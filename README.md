
## **这是一个工作日志：**  


### *2022 0511*   
                    * 我在实验Git; 
                    * 再次尝试，貌似可以随便登录一个人的账号;
                    * 学习解决冲突问题(见git或MMdetection文档);


### *2022 0512*  
                    * 在COCO上训练PVT模型1个epoch,对比参数的增加(参数无明显增加);
                    * Swin模型不能直接在COCO的训练结果上微调;


###  *2022 0513*
                    * 证明Mem可以通过修改Coco_detection.py文件实现图片尺寸(448*448),线程(2)和Batchsize(2)进行修改;
                    * 测试PVT模型是否能在coco数据集上训练的预训练模型上微调(./trainpy);
                    * 将PVT_AtrousLSHTransformer的框架注册到MMdetection,之后用于修改;
                            涉及的文件: 
                            tool/train_Atrous.py  
                            config/pvt/my_pvt_AtrousLSHTransformer.py  
                            mmdet/model/backbone/PVT_AtrousLSHTransformer.py  
                            __init__.py  