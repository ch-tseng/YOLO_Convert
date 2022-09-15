model_path=/DS/Datasets/CH_custom/VOC/Human/Himant_body_parts/trained_weights/YOLOV5sp6/yolov5sp6.pt
new_model_path=/DS/Datasets/CH_custom/VOC/Human/Himant_body_parts/trained_weights/onnx
imgsize=1280

home_python=/home/chtseng/envs/AI/bin/python
home_yolo=/home/chtseng/frameworks/yolov5

if [ ! -d $new_model_path ]
then
    mkdir $new_model_path
fi

cd $home_yolo

baseName=`basename ${model_path}`
extension="${baseName##*.}"
filename="${baseName%.*}"
output_file=${filename}.onnx
output_path="$(dirname "${model_path}")"

$home_python export.py  --imgsz $imgsize --weights $model_path --simplify --iou-thres 0.1 --conf-thres 0.1 --include onnx
echo Output: ${output_path}/${output_file} ${new_model_path}/${filename}_fp32.onnx
mv ${output_path}/${output_file} ${new_model_path}/${filename}_fp32.onnx
