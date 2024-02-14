import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
from importlib import util

def load_file_as_module(location: str):
    """ Function to load module with location path """
    spec = util.spec_from_file_location('', location)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

slider_module = load_file_as_module("./sliding_window_methods/slider.py")

Slider = slider_module.Slider

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.data_name = args.data_name

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, f'log_{self.data_name}_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.outputs = args.outputs
        self.testing_mode = args.testing_mode
        self.window_size = args.window_size
        self.stride = args.stride
        self.sigma = args.sigma

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        # print(G_pred.shape, target.shape)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, f'{self.data_name}_scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, f'{self.data_name}_{self.epoch_acc}.txt'),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _predict_for_window(self, image1, image2, i1, i2, j1, j2):
        window_image1 = image1[:, :, i1:i2, j1:j2]
        window_image2 = image2[:, :, i1:i2, j1:j2]
        return self.net_G(window_image1, window_image2)
    
    def _predict_SW_style(self, image1, image2, model):
        res = model(image1, image2)
        return res
        # res_argmax = res.argmax(1)
        # return res_argmax.unsqueeze(0)

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        # ---------------------------------------------------------
        if self.testing_mode == 'crop':
            height, width = img_in1.shape[2], img_in1.shape[3]
            height_count = height // self.window_size
            width_count = width // self.window_size
            ans = torch.zeros((img_in1.shape[0], 2, height, width)).to(self.device)
            for i in range(height_count):
                for j in range(width_count):
                    i1 = i*self.window_size
                    i2 = (i+1)*self.window_size
                    j1 = j*self.window_size
                    j2 = (j+1)*self.window_size
                    ans[:, :, i1:i2, j1:j2] = self._predict_for_window(img_in1, img_in2, i1, i2, j1, j2)
                if width_count * self.window_size != width:
                    j1 = width - self.window_size
                    j2 = width_count * self.window_size
                    for i in range(height_count):
                        i1 = i*self.window_size
                        i2 = (i+1)*self.window_size
                        ans[:, :, i1:i2, j2:] = self._predict_for_window(img_in1, img_in2, i1, i2, j1, width)[:, :, :, -(width-j2):]
            if height_count * self.window_size != height:
                i1 = height - self.window_size
                i2 = height_count * self.window_size
                for j in range(width_count):
                    j1 = j*self.window_size
                    j2 = (j+1)*self.window_size
                    ans[:, :, i2:, j1:j2] = self._predict_for_window(img_in1, img_in2, i1, height, j1, j2)[:, :, -(height-i2):, :]
                if width_count * self.window_size != width:
                    i1 = height - self.window_size
                    i2 = height_count * self.window_size
                    j1 = width - self.window_size
                    j2 = width_count * self.window_size
                    ans[:, :, i2:, j2:] = self._predict_for_window(img_in1, img_in2, i1, height, j1, width)[:, :, -(height-i2):, -(width-j2):]
            self.G_pred = ans

        elif self.testing_mode == 'sliding_window_avg':
            height, width = img_in1.shape[2], img_in1.shape[3]
            ans = torch.zeros((img_in1.shape[0], 2, height, width)).to(self.device)
            count_matrix = torch.zeros((img_in1.shape[0], 1, height, width)).to(self.device)
            flag0 = ((height - self.window_size) % self.stride) != 0  # maybe %
            flag1 = ((width - self.window_size) % self.stride) != 0
            for i in range(0, height-self.window_size+1, self.stride):
                i_finish = i+self.window_size
                for j in range(0, width-self.window_size+1, self.stride):                    
                    j_finish = j+self.window_size
                    ans[:, :, i:i_finish, j:j_finish] += self._predict_for_window(img_in1, img_in2, i, i_finish, j, j_finish)
                    count_matrix[:, :, i:i_finish, j:j_finish] += 1
                if flag1:
                    j = width - self.window_size
                    ans[:, :, i:i_finish, j:] += self._predict_for_window(img_in1, img_in2, i, i_finish, j, width)
                    count_matrix[:, :, i:i_finish, j:] += 1
            if flag0:
                i = height - self.window_size
                for j in range(0, width-self.window_size+1, self.stride):
                    j_finish = j+self.window_size
                    ans[:, :, i:, j:j_finish] += self._predict_for_window(img_in1, img_in2, i, height, j, j_finish)
                    count_matrix[:, :, i:, j:j_finish] += 1
                if flag1:
                    j = width - self.window_size
                    ans[:, :, i:, j:] += self._predict_for_window(img_in1, img_in2, i, height, j, width)
                    count_matrix[:, :, i:, j:] += 1
            ans /= count_matrix
            self.G_pred = ans
        # TODO do averaging step before argmax
        elif self.testing_mode == 'sliding_window_gauss':
            model = self.net_G
            detect_function = self._predict_SW_style

            slider = Slider(model, detect_function, self.device, self.window_size, self.stride, self.sigma)
            slider.set_generators((img_in1.shape[2], img_in1.shape[3]))
            self.G_pred = slider.predict_for_images_torch(img_in1, img_in2)

        elif self.testing_mode == 'resize':
            self.G_pred = self.net_G(img_in1, img_in2)
        else:
            print("Invalid testing_mode, select from: resize | crop | sliding_window_avg | sliding_window_gauss")
            exit()                                                                                             
        # ---------------------------------------------------------
        
        mask_pred = self.G_pred.detach().argmax(dim=1)[0]
        mask_pred_numpy = mask_pred.data.cpu().numpy()
        plt.imsave(os.path.join(self.outputs, batch['name'][0]), mask_pred_numpy*255)
        # print(mask_pred_numpy.max())

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            # print(batch['A'].shape)
            with torch.no_grad():
                self._forward_pass(batch)
            if 'L' in self.batch:
                self._collect_running_batch_states()
        if 'L' in self.batch:
            self._collect_epoch_states()
