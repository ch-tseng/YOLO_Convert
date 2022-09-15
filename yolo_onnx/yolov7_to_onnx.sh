model_path=/DS/Datasets/CH_custom/VOC/Human/Face_mask_eyeball/trained_weights/yolov7e6/yolov7e6.pt
new_model_path=/DS/Datasets/CH_custom/VOC/Human/Face_mask_eyeball/trained_weights/onnx
imgsize=1280

home_python=/home/chtseng/envs/AI/bin/python
home_yolo=/home/chtseng/frameworks/yolov7

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

$home_python export.py --weights $model_path --grid --end2end --simplify \
        --img-size $imgsize $imgsize --max-wh $imgsize
mv ${output_path}/${output_file} ${new_model_path}/${filename}_fp32.onnx
