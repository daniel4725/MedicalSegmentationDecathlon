import wandb
# https://docs.wandb.ai/guides/sweeps/configuration

sweep_name = "heart aug sweep"
sweep_config = {
    "method": "grid",
    "name": sweep_name,
    "metric": {
        "goal": "maximize",
        "name": "val/Dice_target_smooth"
        },
    "parameters": {
        # ------ static parameters -------
        'sweep_name': {
            'value': sweep_name
            },
        'load2ram': {
            'value': True
            },
        'checkpoint_dir': {
            'value': "/media/rrtammyfs/Users/daniel/DeepVisionProject/checkpoints"
            },
        'save_cp_every': {
            'value': 10000
            },
        'continue_from_checkpoint': {
            'value': "None"
            },
        'cp_epoch': {
            'value': "None"
            }
        }
}
# ------ changable parameters -------
changeable_parameters = {
    'epochs': {
        'value': 500
        },
    'GPU': {
        'value': 0
        },
    # 'class_weights': {
    #     'value': [1]  # if len(class_weights) <= 1 will calculate the weights from the data
    #     },
    'data_type': {
        'value': "Heart"   # Hippocampus  Heart
        },
    'data_fold': {
        'value': 0  # 0 to 4
    }

}
sweep_config["parameters"].update(changeable_parameters)

# ------ sweepable parameters -------
sweep_parameters = {
    'repeat': {
        'values': [1, 2, 3]  # variable to repeat all while grid searching
    },
    'init_features': {
        'values': [32]
        # 'value': 32
    },
    'transform': {
        'values': ["None", "heart_noise_aug", "heart_flip_aug", "heart_rescale5prct_aug", "heart_rescale10prct_aug"]
    },
    'L2': {
        'values': [0.025]
    },
    'dropout': {
        'values': [0.0, 0.2]
        # 'values': [0.0, 0.1]
    },
    'lr': {
        'values': [0.0001]
        # 'values': [0.01, 0.1]
    },
    'dice_factor': {
        'values': [0.75]
    },
    'batch_size': {
        'values': [2]
        # 'values': [1]
        # 'value': 4
    },
    'class_weights': {
        # 'value': [0.52, 14]
        # 'value': [0.5, 124]   # the weights in the full images
        'value': [0.5, 68]   # the weights in the full images without the black stripes (up and down)
        # 'value': [0]  # calculate by itself
    },
    'model': {
        # 'values': ["UNet", "UNet_Short"]
        'value': "UNet"
    },
    'heart_crop_prob': {
        # 'values': [0.0, 0.2]
        'values': [0.0]
    }

}
sweep_config["parameters"].update(sweep_parameters)

# [ 0.5215, 12.1419]
command = ["/usr/bin/env", "python3.6", "train_models.py", "--sweeping", "--runfromshell"]
sweep_config["command"] = command

sweep_id = wandb.sweep(sweep_config, project="DeepVisionProject-MSD")
print("run the following commands in the terminal for multiple GPU agents:")
print(f"{sweep_name}:")
for i in range(4):
    print(f"CUDA_VISIBLE_DEVICES={i} python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/{sweep_id}")

"""
heart init sweep 1:
CUDA_VISIBLE_DEVICES=0 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/gmvspf3z
CUDA_VISIBLE_DEVICES=1 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/gmvspf3z
CUDA_VISIBLE_DEVICES=2 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/gmvspf3z
CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/gmvspf3z

heart normal sweep:
CUDA_VISIBLE_DEVICES=0 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/qy8urr8s
CUDA_VISIBLE_DEVICES=1 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/qy8urr8s
CUDA_VISIBLE_DEVICES=2 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/qy8urr8s
CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/qy8urr8s

heart aug sweep:
CUDA_VISIBLE_DEVICES=0 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/urvxlll7
CUDA_VISIBLE_DEVICES=1 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/urvxlll7
CUDA_VISIBLE_DEVICES=2 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/urvxlll7
CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/urvxlll7

function set-title() {
  if [[ -z "$ORIG" ]]; then
    ORIG=$PS1
  fi
  TITLE="\[\e]2;$*\a\]"
  PS1=${ORIG}${TITLE}
}
set-title 'GPU 3'

clear

"""


