# adapted from PaletteNeRF
# https://github.com/zfkuang/PaletteNeRF

CONFIGFILE=$1;
shift

if [ $# -eq 0 ]; then
    echo "Error: a config file is required."
    exit
fi
if [ ! -f "$CONFIGFILE" ]; then
    echo "Error: $CONFIGFILE does not exist."
    exit
fi
source $CONFIGFILE

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      model="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

ts=$(date +%s)

if [[ $model == 'nerf' ]]; then
    python main_nerf.py \
    $data_dir \
    --workspace ${workspace} \
    --iters ${iters} \
    --bound ${bound} \
    --offset ${offset} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    -O \
    --dt_gamma 0
elif [[ $model == 'recolor' ]]; then
    python main_nerf.py \
    $data_dir \
    --workspace ${workspace} \
    --iters ${iters} \
    --bound ${bound} \
    --offset ${offset} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    -O \
    --dt_gamma 0 \
    --gui \
    --train_steps_style 10000 \
    --train_steps_distill 7000 \
    --weight_loss_non_uniform 1e-7 \
    --offset_loss 5e-5 \
    --palette_loss_valid 1 \
    --num_palette_bases 8 \
    --ablation_dir test \
    --ablation_folder ${name}_'recolor'_${ts} \
    --smooth_trans_weight 1e-3
elif [[ $model == 'style' ]]; then
    python main_nerf.py \
    $data_dir \
    --workspace ${workspace} \
    --iters ${iters} \
    --bound ${bound} \
    --offset ${offset} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    -O \
    --dt_gamma 0 \
    --gui \
    --train_steps_style 10000 \
    --train_steps_distill 7000 \
    --weight_loss_non_uniform 1e-7 \
    --offset_loss 5e-5 \
    --palette_loss_valid 1 \
    --num_palette_bases 8 \
    --ablation_dir test \
    --ablation_folder ${name}_'style'_${ts} \
    --smooth_trans_weight 1e-3 \
    --tv_weight 1e-4 \
    --tv_depth_guide \
    --depth_disc_weight 5e-4 \
     --style_weight 1.3e2 \
     --style_layers 10 \
     --style_layers 12 \
     --style_layers 14 --style_image wave_style.png
else
    echo "Invalid model. Options are: nerf, recolor, style"
fi
