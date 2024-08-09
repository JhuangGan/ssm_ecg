# 实现目的，将不同数据载入，变成这里裁剪后的，使用memmap的dataloader
# 第一步，变成npy，并进行memmap变换，得到相关数据
# 先把这里的其他变化，整理成册先
# 第二步，其他数据，进行memmap变换，然后变成timeseries的dataloader

# 第二步超级麻烦，需要memmap.npy, memmap_meta.npz, df, lbl_itos也要，
# 还是先把这些文件的具体格式和变换，整理清楚先

#####################第一步#############################
df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=1, target_fs=target_fs, 
                                                                           channels=12, channel_stoi=channel_stoi_default, 
                                                                           target_folder=target_folder_ptb_xl, recreate_data=True)
# 一些新的标签'label_all', 'label_diag', 'label_form', 'label_rhythm',
    #    'label_diag_subclass', 'label_diag_superclass', 'dataset',
    #    'label_all_filtered', 'label_all_filtered_numeric', 'label_all_numeric',
    #    'label_diag_filtered', 'label_diag_filtered_numeric',
    #    'label_diag_numeric', 'label_form_filtered',
    #    'label_form_filtered_numeric', 'label_form_numeric',
    #    'label_rhythm_filtered', 'label_rhythm_filtered_numeric',
    #    'label_rhythm_numeric', 'label_diag_subclass_filtered',
    #    'label_diag_subclass_filtered_numeric', 'label_diag_subclass_numeric',
    #    'label_diag_superclass_filtered',
    #    'label_diag_superclass_filtered_numeric',
    #    'label_diag_superclass_numeric', 'data', 'data_mean', 'data_std',
    #    'data_length']  

# ["dataset"]="ptb_xl"

# data是npy的文件路径，data_mean是该文件的mean，std同理，data_length是


df.to_pickle(target_root/("df"+filename_postfix+".pkl"), protocol=protocol)




lbl_itos_ptb_xl={'label_all_filtered': array(['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI',
       'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT',
       'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN',
       'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN',
       'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB',
       'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT',
       'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)',
       'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD',
       'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_',
       'TRIGU', 'VCLVH', 'WPW'], dtype='<U7'), 'label_all': array(['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI',
       'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT',
       'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN',
       'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN',
       'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB',
       'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT',
       'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)',
       'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD',
       'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_',
       'TRIGU', 'VCLVH', 'WPW'], dtype='<U7'), 'label_diag_filtered': array(['1AVB', '2AVB', '3AVB', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'CLBBB',
       'CRBBB', 'DIG', 'EL', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS',
       'INJIL', 'INJIN', 'INJLA', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
       'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD',
       'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LPFB', 'LVH', 'NDT', 'NORM',
       'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'WPW'], dtype='<U7'), 'label_diag': array(['1AVB', '2AVB', '3AVB', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'CLBBB',
       'CRBBB', 'DIG', 'EL', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS',
       'INJIL', 'INJIN', 'INJLA', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL',
       'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD',
       'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LPFB', 'LVH', 'NDT', 'NORM',
       'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'WPW'], dtype='<U7'), 'label_form_filtered': array(['ABQRS', 'DIG', 'HVOLT', 'INVT', 'LNGQT', 'LOWT', 'LPR', 'LVOLT',
       'NDT', 'NST_', 'NT_', 'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'STD_',
       'STE_', 'TAB_', 'VCLVH'], dtype='<U6'), 'label_form': array(['ABQRS', 'DIG', 'HVOLT', 'INVT', 'LNGQT', 'LOWT', 'LPR', 'LVOLT',
       'NDT', 'NST_', 'NT_', 'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'STD_',
       'STE_', 'TAB_', 'VCLVH'], dtype='<U6'), 'label_rhythm_filtered': array(['AFIB', 'AFLT', 'BIGU', 'PACE', 'PSVT', 'SARRH', 'SBRAD', 'SR',
       'STACH', 'SVARR', 'SVTAC', 'TRIGU'], dtype='<U5'), 'label_rhythm': array(['AFIB', 'AFLT', 'BIGU', 'PACE', 'PSVT', 'SARRH', 'SBRAD', 'SR',
       'STACH', 'SVARR', 'SVTAC', 'TRIGU'], dtype='<U5'), 'label_diag_subclass_filtered': array(['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI',
       'ISC_', 'IVCD', 'LAFB/LPFB', 'LAO/LAE', 'LMI', 'LVH', 'NORM',
       'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW', '_AVB'],
      dtype='<U9'), 'label_diag_subclass': array(['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI',
       'ISC_', 'IVCD', 'LAFB/LPFB', 'LAO/LAE', 'LMI', 'LVH', 'NORM',
       'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW', '_AVB'],
      dtype='<U9'), 'label_diag_superclass_filtered': array(['CD', 'HYP', 'MI', 'NORM', 'STTC'], dtype='<U4'), 'label_diag_superclass': array(['CD', 'HYP', 'MI', 'NORM', 'STTC'], dtype='<U4')}


if(isinstance(lbl_itos,dict)):#dict as pickle
        outfile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "wb")
        pickle.dump(lbl_itos, outfile, protocol=protocol)
        outfile.close()
    else:#array
        np.save(target_root/("lbl_itos"+filename_postfix+".npy"),lbl_itos)



# 转成 npy
# 直接读取保存为npy
np.save(target_root_ptb_xl/(filename.stem+".npy"),data)


# memmap的对照，

reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=False)

########################第二步
# 将memmap.npy和meta，转成为timeseries data



###################################################

class ECGDataSetWrapper(object):
    def __init__(self, batch_size, num_workers, target_folder, input_size=250, label="label_diag_superclass", test=False,
                 shuffle_train=True, drop_last=True, nomemmap=False, test_folds=[8, 9], filter_label=None, combination='both', val_stride=None,
                  normalize=False, use_meta_information_in_head=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_folder = Path(target_folder)
        self.input_size = input_size
        self.val_ds_idmap = None
        self.lbl_itos = None
        self.train_ds_size = 0
        self.val_ds_size = 0
        self.label = label
        self.test = test
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        # only for training without memmap file (e.g. syn vs real)
        self.no_memmap = nomemmap
        self.test_folds = np.array(test_folds)
        self.filter_label = filter_label
        self.combination = combination
        self.val_stride = val_stride
        self.normalize=normalize
        self.use_meta_information_in_head=use_meta_information_in_head

    def get_data_loaders(self):
        if self.normalize:
            print("use normalizaiton")
            data_augment = transforms.Compose([ToTensor(), TNormalize()])
        else:
            data_augment = ToTensor()

        if self.no_memmap:
            train_ds, val_ds = self._get_datasets_no_memmap(
                self.target_folder, transforms=data_augment)
        else:
            train_ds, val_ds = self._get_datasets(
                self.target_folder, transforms=data_augment)
        self.val_ds = val_ds
        self.val_ds_idmap = val_ds.get_id_mapping()

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  num_workers=self.num_workers, pin_memory=True, shuffle=self.shuffle_train, drop_last=self.drop_last)
        valid_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers, pin_memory=True)

        self.train_ds_size = len(train_ds)
        self.val_ds_size = len(val_ds)
        return train_loader, valid_loader

    def get_training_params(self):
        chunkify_train = False
        chunkify_valid = True
        chunk_length_train = self.input_size  # target_fs*6
        chunk_length_valid = self.input_size
        min_chunk_length = self.input_size  # chunk_length
        stride_length_train = chunk_length_train//4  # chunk_length_train//8
        stride_length_valid = self.val_stride if self.val_stride is not None else self.input_size #//2  # chunk_length_valid

        copies_valid = 0  # >0 should only be used with chunkify_valid=False
        return chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid

    def get_folds(self, target_folder):
        if self.test:
            valid_fold = 10
            test_fold = 9
        else:
            valid_fold = 9
            test_fold = 10
        if "thew" in str(target_folder) or "chapman" in str(target_folder) or "sph" in str(target_folder) or 'icbeb' in str(target_folder):
            valid_fold -= 1
            test_fold -= 1

        train_folds = []
        train_folds = list(range(1, 11))
        train_folds.remove(test_fold)
        train_folds.remove(valid_fold)
        train_folds = np.array(train_folds)
        train_folds = train_folds - \
            1 if "thew" in str(target_folder) or "zheng" in str(
                target_folder) else train_folds
        return train_folds, valid_fold, test_fold

    def get_dfs(self, df_mapped, target_folder):
        train_folds, valid_fold, test_fold = self.get_folds(target_folder)
        df_train = df_mapped[(df_mapped.strat_fold.apply(
            lambda x: x in train_folds))]
        df_valid = df_mapped[(df_mapped.strat_fold == valid_fold)]
        df_test = df_mapped[(df_mapped.strat_fold == test_fold)]
        return df_train, df_valid, df_test

    def _get_datasets(self, target_folder, transforms=None):
        logger.info("get dataset from " + str(target_folder))
        chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid = self.get_training_params()

        ############### Load dataframe with memmap indices ##################

        df_mapped, lbl_itos, mean, std = load_dataset(target_folder)
        self.lbl_itos = lbl_itos

        ############### get right labels & map to multihot encoding ###################
        if "ptb" in str(target_folder):
            label = self.label  # just possible for ptb xl
            self.lbl_itos = np.array(lbl_itos[label])
            label = label + "_filtered_numeric"
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: multihot_encode(x, len(self.lbl_itos)))
        elif 'sph' in str(target_folder):
            label = self.label  # just possible for ptb xl
            self.lbl_itos = np.array(lbl_itos[label])
            label = label + "_numeric"
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: multihot_encode(x, len(self.lbl_itos)))
            
        elif "chapman" in str(target_folder):
            label = self.label
            # self.lbl_itos = np.array(lbl_itos[label.split("_")[-1]])
            self.lbl_itos = np.array(lbl_itos[label])
            df_mapped["label"] = df_mapped[label + "_numeric"].apply(
                lambda x: multihot_encode(x, len(self.lbl_itos)))
        elif "cinc" in str(target_folder):
            label = 'label'
            self.lbl_itos = lbl_itos
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos)))
        elif 'icbeb' in str(target_folder):
            label = self.label
            self.lbl_itos = np.array(lbl_itos[label])
            df_mapped['label'] = df_mapped[label].apply(
                    lambda x: multihot_encode(x, len(self.lbl_itos)))
        else:
            label = "label"
            self.lbl_itos = lbl_itos
            df_mapped["label"] = df_mapped[label].apply(
                lambda x: np.array([1, 0, 0, 0, 0]))

        self.num_classes = len(self.lbl_itos)
        # import pdb; pdb.set_trace()
        df_mapped["diag_label"] = df_mapped[label].copy()

        df_train, df_valid, df_test = self.get_dfs(df_mapped, target_folder)

        cols_static = ['sex', 'age_nonan', 'height_nonan', 'weight_nonan', 'age_isnan', 'height_isnan', 
                     'weight_isnan'] if self.use_meta_information_in_head else None
                        
        ################## create datasets ########################
        train_ds = TimeseriesDatasetCrops(df_train, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_train if chunkify_train else 0, cols_static=cols_static,
                                          min_chunk_length=min_chunk_length, stride=stride_length_train, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/("memmap.npy"))
        val_ds = TimeseriesDatasetCrops(df_valid, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_valid if chunkify_valid else 0, cols_static=cols_static,
                                        min_chunk_length=min_chunk_length, stride=stride_length_valid, transforms=transforms, annotation=False, col_lbl="label", memmap_filename=target_folder/("memmap.npy"))

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        return train_ds, val_ds

    def _get_datasets_no_memmap(self, target_folder, transforms=None):
        logger.info("get dataset from " + str(target_folder))
        chunkify_train, chunkify_valid, chunk_length_train, chunk_length_valid, min_chunk_length, stride_length_train, stride_length_valid, copies_valid = self.get_training_params()

        ############### Load dataframe with memmap indices ##################
        
        df = pickle.load(open(target_folder/("df.pkl"), 'rb'))
        # df = pd.read_pickle(target_folder/'df.pkl')
        lbl_itos = pickle.load(
            open(target_folder/("lbl_itos.pkl"), 'rb'))[self.label]

        self.lbl_itos = lbl_itos

        self.num_classes = len(self.lbl_itos)

        train_folds = []
        train_folds = list(range(10))

        for fold in self.test_folds:
            train_folds.remove(fold)
        train_folds = np.array(train_folds)

        df_train = df[(df.strat_fold.apply(lambda x: x in train_folds))]
        df_test = df[(df.strat_fold.apply(lambda x: x in self.test_folds))]
        patho_label_to_numeric = {"AVBlock":0, "LBBB": 1, "Normal":2, "RBBB":3}

        def filter_dataset(data_df, label=None, combination=''):
            if combination == 'real':
                data_df = data_df[data_df['label_real'].apply(
                    lambda x: x.argmax() == 0)]
            elif combination == 'syn':
                data_df = data_df[data_df['label_real'].apply(
                    lambda x: x.argmax() == 1)]

            if label is not None:
                data_df = data_df[data_df['label_patho'].apply(
                    lambda x: x.argmax() == patho_label_to_numeric[label])]

            return data_df

        
        df_train = filter_dataset(
            df_train, self.filter_label, combination=self.combination.split("_")[0])
        
        df_test = filter_dataset(
            df_test, self.filter_label, combination=self.combination.split("_")[1] if "_" in self.combination else '')
        
        self.df_train = df_train
        self.df_test = df_test
        
        ################## create datasets ########################
        train_ds = TimeseriesDatasetCrops(df_train, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_train if chunkify_train else 0,
                                          min_chunk_length=min_chunk_length, stride=stride_length_train, transforms=transforms, annotation=False, col_lbl=self.label)
        test_ds = TimeseriesDatasetCrops(df_test, self.input_size, num_classes=self.num_classes, data_folder=target_folder, chunk_length=chunk_length_valid if chunkify_valid else 0,
                                         min_chunk_length=min_chunk_length, stride=stride_length_valid, transforms=transforms, annotation=False, col_lbl=self.label)

        return train_ds, test_ds

# target_folder = 

dataset = ECGDataSetWrapper(
        batch_size, num_workers, target_folder, test=test, label=label_class, 
        input_size=input_size, shuffle_train=shuffle_train, nomemmap=nomemmap,
        normalize=normalize, test_folds=test_folds, combination=combination,
        filter_label=filter_label, val_stride=val_stride, use_meta_information_in_head=use_meta_information_in_head)

train_loader, valid_loader = dataset.get_data_loaders()