
## **这是一个工作日志：**  


### *2022 0511*   
                * 我在实验Git; 
                * 再次尝试，貌似可以随便登录一个人的账号;
                * 学习解决冲突问题(见git或MMdetection文档);


### *2022 0512*  
                * 在COCO上训练PVT模型1个epoch,对比参数的增加(参数无明显增加);
                * Swin模型不能直接在COCO的训练结果上微调;


###  *2022 0513*
                * 证明Mem可以通过修改Coco_detection.py文件实现图片尺寸(448*448),线程(2)和Batchsize(2)进行修改(也许这样一来,原本不能使用的模型就能用了.); 
                * 测试PVT模型是否能在coco数据集上训练的预训练模型上微调(./trainpy);
                * 将PVT_AtrousLSHTransformer的框架注册到MMdetection,之后用于修改;
                        涉及的文件: 
                        tool/train_Atrous.py  
                        config/pvt/my_pvt_AtrousLSHTransformer.py  
                        mmdet/model/backbone/PVT_AtrousLSHTransformer.py  
                        __init__.py  

###  *2022 0514*
                * 证明MMdetection与darknet存在不同,在mmdetection的config文件下设置的epoch不会受到预训练模型的干扰;
                * MMdetection提供了很多有趣的检测和测量方法,例如log曲线的分析 \ 结果分析 \ 模型和数据集可视化 \ 数据集类别错误分析 \ 模型转换 \ 模型复杂度计算 \ 数据集转换 \ FPS计算 \  度量计算 \ 优化检测框 \ 混合矩阵计算

###  *2022 0516*
                * PVT v1代码梳理

###  *2022 0517*
![img](https://img-blog.csdnimg.cn/20210309175435338.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L29ZZVpob3U=,size_16,color_FFFFFF,t_70)

                * 修改结构
                * 检测


###  *2022 0519*
                * 修改无空间缩减注意力


###  *2022 0521*
                * 修改图像尺寸,尽在最开始进行图片的嵌入和位置的嵌入


###  *2022 0523*
                * 原始网络的情况:
                        PVT: [ [patch, [pos_embed,layer1,layer2], norm],[patch, [pos_embed,layer1,layer2], norm],[patch, [pos_embed,layer1,layer2], norm], [patch, [pos_embed,layer1,layer2], norm], ]
                * 目前网络的情况:
                         patchSize大小为5,Stride大小为5,保证不重叠的进行图片嵌入;
                         图片的特征和位置嵌入只在输入网络前进行一次;
                        Atrous: [ patch, pos_embed, [PVT, norm], [PVT, norm], [PVT, norm], [PVT, norm] ]
                * 计划修改网络:
                        [  [[patch, pos], [[layer1,layer2], norm]],  
                            [[reshape,linear], [[layer1,layer2], norm]], 
                            [[reshape,linear], [[layer1,layer2], norm]], 
                            [[reshape,linear], [[layer1,layer2], norm]] ]


###  *2022 0524*
                * 目前网络的特征图尺寸不变,PatchSize不变.因为删除了空间注意并修改了特征图尺寸,导致模型的Memery变大,不得不减小图片的尺寸为(224*224),或者降低BatchSize

                * 下一步:检查各环节是否存在问题,计划将注意力进行修改(可以参考利用膨胀卷积的部分)


###   *2022 0525*
                * 修改计算方式,让其可以运行


###  *2022 0526*
                * 测试github的reset功能


###  *2022 0527*
                * 整理代码


###  *2022 0530*
                * 测试修改的无特征金字塔和空间缩减注意力的代码,并训练之.
                * 查找Atrous self attention相关代码实现
                
[Generating Long Sequences with Sparse Transformers](https://paperswithcode.com/paper/190410509)

[VideoGPT](https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/attention.py)

[torch-blocksparse](https://github.com/ptillet/torch-blocksparse/blob/master/torch_blocksparse/deepspeedsparseselfattention.py)

[Keras-attention](https://github.com/bojone/attention/blob/master/attention_keras.py)

[VideoGPT](https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/attention.py)

[torch-blocksparse](https://github.com/ptillet/torch-blocksparse/blob/master/torch_blocksparse/deepspeedsparseselfattention.py)

[Keras-attention](https://github.com/bojone/attention/blob/master/attention_keras.py)


###  *2022 0617*
                * workdir/my_pvt_AtrousLSHTrasnformer
                        此文件是不采用空间缩减注意力(即采用全注意力)的非FPN架构pvt v1模型;
                        训练了12epoch
                        [需要对其实验结果进行测试,测试在COCO下的map,IOU,fps,flops等;最好实现Attention的可视化.]
                * workdir/retinanet_pvt-t_fpn_1x_coc
                        此文件是基准pvt v1训练后的结果,同样需要进行测试与验证,作为对照组.


###  *2022 0618*
                *测试workdir/my_pvt_AtrousLSHTrasnformer和workdir/retinanet_pvt-t_fpn_1x_coc的实验数据;
                *设置软链接,链接VOC格式的自定义数据集.


###  *2022 0618*
                *终端输出的东西在mmdet/datasets/pipline/loading.py下改.


###  *2022 0621*
                *制作自定义的数据集,训练之


###  *2022 0622*
                *测试结果:workdir/my_pvt_AtrousLSHTrasnformer_voc和workdir/retinanet_pvt-t_fpn_1x_coc_voc的实验数据;
                *重新测试workdir/my_pvt_AtrousLSHTrasnformer和workdir/retinanet_pvt-t_fpn_1x_coc的图像实验数据.
                [制定GPU] python tools/test.py [config] [weights] --out [save_dir, 必须制定文件格式哦,写为xxx/results.pkl] --show [show_images] --show-dir [image_save_dir]

                eg: CUDA_VISIBLE_DEVICES="2" python tools/test.py work_dirs/retinanet_pvt-t_fpn_1x_coco/retinanet_pvt-t_fpn_1x_coco.py work_dirs/retinanet_pvt-t_fpn_1x_coco/epoch_12.pth --out results/retinanet_pvt-t_fpn_1x_coco/results.pkl --show-dir results/retinanet_pvt-t_fpn_1x_coco


###  *2022 0623*
                *目标:画出自定义数据集的融合矩阵,并遍历保存测试图片.
                

###  *2022 0625*
                *整理查找Atrous self attention相关代码实现,梳理代码结构


###  *2022 0630*
                *完成代码梳理,参考VideoGPT进行代码修改,使用其设计的稀疏注意力


###  *2022 0712*
                *选择添加稀疏注意力机制

###  *2022 0815*
                在PVT_Atrous下进行修改,利用膨胀卷积降低参数量,并保证每一阶段都是FPN的合适输出.              
                

###  *2022 0829*
                * 原始网络的情况:
                        PVT: [ [patch, [pos_embed,layer1,layer2], norm],[patch, [pos_embed,layer1,layer2], norm],[patch, [pos_embed,layer1,layer2], norm], [patch, [pos_embed,layer1,layer2], norm], ]
                
                * 计划修改网络:
                        [  [[patch, pos], [[layer1,layer2], norm]],  
                            [[reshape,linear], [[layer1,layer2], norm]], 
                            [[reshape,linear], [[layer1,layer2], norm]], 
                            [[reshape,linear], [[layer1,layer2], norm]] ]
                
                存在问题:模型依旧不收敛,怀疑是没有进行位置嵌入或直接线性映射不可取.
                从根源进行替换: 
                        1.  恢复整体的网络结构PVT, 取消patchEmbed的步长和尺度,进行1*1conv(尝试),看看是否会收敛.

                        2.  减少进入FPN的参数量,利用膨胀卷积进行构造,但保证骨干网络中计算的参数量不变.