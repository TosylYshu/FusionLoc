# FusionLoc
I have graduated from university and become a social being. There is no time for me to rearrange the disorganized codes, so I have no intention to release the codes and checkpoints temporarily. 
## Dataset
We release the dataset which is used on our paper. You may download our dataset at:
**Google Drive:**[https://drive.google.com/drive/folders/1y3XTs_L8HmOpIU2b33_eNQ_O-XsdNfiR](https://drive.google.com/drive/folders/1y3XTs_L8HmOpIU2b33_eNQ_O-XsdNfiR)
## Example Code for Quick Start
`main.py` contains the majority of the code, and has two different modes (`train`, `test`) .
### Train
A model can be trained using (the following flags):
```shell
python3 main.py --mode train --dataset_path /path/to/3Floor-datasets/ --labels_file ./datasets/3Floor/abs_pose.csv_corridor_train.csv --rssi_file ./datasets/3Floor/rssi.csv_corridor_train_dense.csv
```
### Test
To test a previously trained model (replace directory with correct dir for your case):
```shell
python3 main.py --mode test --dataset_path /path/to/3Floor-datasets/ --labels_file ./datasets/3Floor/abs_pose.csv_corridor_test.csv --rssi_file ./datasets/3Floor/rssi.csv_corridor_test_dense.csv --checkpoint_path <path>
```
## Citation
```bibtex
@article{tang2024novel,
  title={A Novel Multi-Modal Feature-Level Fusion Scheme for High Accurate Indoor Localization},
  author={Tang, Siyu and Huang, Kaixuan and Zhang, Shunqing},
  journal={IEEE Sensors Journal},
  year={2024},
  publisher={IEEE}
}

```
```bibtex
@inproceedings{tang2023hybrid,
  title={Hybrid Cascaded and Feature-Level Fusion Scheme for Multi-Modal Indoor Localization},
  author={Tang, Siyu and Huang, Kaixuan and Zhang, Shunqing},
  booktitle={2023 IEEE 97th Vehicular Technology Conference (VTC2023-Spring)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}

```
## Acknowledgments
Code is inspired by [multi-scene-pose-transformer](https://github.com/yolish/multi-scene-pose-transformer). 