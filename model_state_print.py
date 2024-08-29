
# 用于打印S4模型的具体参数机构
import torch
checkpoint_path = 

pl_model = ECGLightningModel(
        fun,
        args.batch_size,
        datamodule.num_samples,
        lr=args.lr,
        rate=args.rate,
        loss_fn=nn.CrossEntropyLoss(
        ) if args.binary_classification else F.binary_cross_entropy_with_logits,
        use_meta_information_in_head=args.use_meta_information_in_head,
        dataset_name=str(os.path.basename(args.target_folder)),
        input_size_name=str(args.input_size),
        label=str(args.label_class),
        add_info=str(args.add_info)
    )

print()