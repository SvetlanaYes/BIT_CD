
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = '../datasets/LEVIR_CD_dataset_256/' # 'path to the root of LEVIR-CD dataset'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        elif data_name == 'LEVIR_CD_dataset':
            self.root_dir = '../datasets/LEVIR_CD_dataset/'
        elif data_name == 'LEVIR_CD_dataset_256':
            self.root_dir = '../datasets/LEVIR_CD_dataset_256/'
        elif data_name == 'data_256':
            self.root_dir = '../datasets/data_256/'
        elif data_name == 'data_512':
            self.root_dir = '../datasets/data_512/'
        elif data_name == 'google_earth_pro':
            self.root_dir = '../datasets/google_earth_pro2_256/'
        elif data_name == 'CDD_256':
            self.root_dir = '../datasets/CDD_256/'
        elif data_name == 'S2Looking':
            self.root_dir = '../datasets/S2Looking/'
        elif data_name == 'SYSU_CD_256':
            self.root_dir = '../datasets/SYSU_CD_256'
        elif data_name == 'merged_all_5_originals':
            self.root_dir = '../datasets/merged_all_5_originals'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

