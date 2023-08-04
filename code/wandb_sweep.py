import wandb
# https://docs.wandb.ai/guides/sweeps/configuration

sweep_name = "hippocampus reg sweep"
sweep_config = {
    "method": "bayes",
    "name": sweep_name,
    "metric": {
        "goal": "maximize",
        "name": "val/Dice_target"
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
        'value': 400
        },
    'GPU': {
        'value': 0
        },
    # 'class_weights': {
    #     'value': [1]  # if len(class_weights) <= 1 will calculate the weights from the data
    #     },
    'data_type': {
        'value': "Hippocampus"   # Hippocampus  Heart
        },
    'data_fold': {
        'value': 0   # 0 to 4
        },
    'transform': {
        'value': "None"
        }

}
sweep_config["parameters"].update(changeable_parameters)

# ------ sweepable parameters -------
sweep_parameters = {
    'init_features': {
        'values': [8, 16, 32]
    },
    'L2': {
        'values': [0.1, 0.05, 0.01]
    },
    'dropout': {
        'values': [0.0, 0.2, 0.4]
    },
    'lr': {
        'values': [0.01, 0.005, 0.0005]
    },
    'dice_factor': {
        'values': [0, 0.33, 0.66, 1]
    },
    'batch_size': {
        'values': [4, 8]
    },
    'class_weights': {
        'values': [[0., 1, 1], [0.1, 12, 12], [0.3526, 11.7466, 12.6686]]
    },
    'model': {
        'values': ["UNet", "UNet_Short"]
    }
}
sweep_config["parameters"].update(sweep_parameters)


command = ["/usr/bin/env", "python3.6", "train_models.py", "--sweeping", "--runfromshell"]
sweep_config["command"] = command

sweep_id = wandb.sweep(sweep_config, project="DeepVisionProject-MSD")
print("run the following commands in the terminal for multiple GPU agents:")
for i in range(4):
    print(f"CUDA_VISIBLE_DEVICES={i} python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/{sweep_id}")

"""
CUDA_VISIBLE_DEVICES=0 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/kivrt3nh
CUDA_VISIBLE_DEVICES=1 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/kivrt3nh
CUDA_VISIBLE_DEVICES=2 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/kivrt3nh
CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/kivrt3nh

hippocampus init sweep:
CUDA_VISIBLE_DEVICES=0 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/aa799v20
CUDA_VISIBLE_DEVICES=1 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/aa799v20
CUDA_VISIBLE_DEVICES=2 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/aa799v20
CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/aa799v20

hippocampus augment sweep:
CUDA_VISIBLE_DEVICES=0 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/zlhfvn9a
CUDA_VISIBLE_DEVICES=1 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/zlhfvn9a
CUDA_VISIBLE_DEVICES=2 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/zlhfvn9a
CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/zlhfvn9a

hippocampus final sweep:
CUDA_VISIBLE_DEVICES=0 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/feffe0rq
CUDA_VISIBLE_DEVICES=1 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/feffe0rq
CUDA_VISIBLE_DEVICES=2 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/feffe0rq
CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/feffe0rq

CUDA_VISIBLE_DEVICES=3 python3.6 -m wandb agent team_dd/DeepVisionProject-MSD/qc5tlfpx

function set-title() {
  if [[ -z "$ORIG" ]]; then
    ORIG=$PS1
  fi
  TITLE="\[\e]2;$*\a\]"
  PS1=${ORIG}${TITLE}
}
set-title 'GPU 3'

cd /home/duenias/PycharmProjects/deepvision
clear


"""


