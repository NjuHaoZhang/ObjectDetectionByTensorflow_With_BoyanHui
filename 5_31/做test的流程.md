1. 制作pb文件 {模型数据持久化}
cd 到tensorflow ... api的 object_detection目录下 {demo也是放在这个目录下，和之前一样}
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path="./demo/faster_rcnn_resnet101_voc07.config" \
    --trained_checkpoint_prefix="./demo/train/model.ckpt-xxxxx" \        
    --output_directory="./demo/train/pb/"

注：	
model.ckpt-xxxxx是指你最好的那一个checkpoint
	
	
	
	
2. 重开一个专用于test的项目

2.1 项目的结构 {我会发截图加文字进行说明}
data 里面放 pascal_label_map.pbtxt {从之前的demo中copy过来}
my_models 放 frozen_inference_graph.pb {从上一步生成的pb复制过来}
test_images 放测试图片，格式只能为  xxxx.jpeg {我已经按照test的20%规格，挑选了20%，并整理好}
object_detection 就是tensorflow ... api的 .../research/object_detection目录复制过来 {之前配置好的demo项目中的那个}
{之所以写这么麻烦，是为了适应后面代码的改动，以后如果加新东西就很容易扩展}

2.2 源代码ImageTest.py的修改 {见我的中文注释}
2.3 python ImageTest.py {运行它} 




3. accuracy及speed，我还要想一下怎么拿出来 {24点前出方案}



4. 现在这个项目组织有点混乱，我昨天晚上已经开始优化了，等下次做1000张图片的project及视频检测时，我会把所有功能都整合到一个项目中去
5. 还有两个小问题，理论上应该是能有预期的效果。但是我电脑上还没跑10s内存就崩了。所以师兄如果你发现了异常，反馈给我，我验证下自己的想法