# CADET

Code for the paper "[CADET: Calibrated Anomaly Detection for Mitigating Hardness Bias](https://www.ijcai.org/proceedings/2022/0278.pdf)"

## Demo Visualization
We have provided a juypter notebook `demo.ipynb`[(link)](https://github.com/d-ailin/CADET/blob/main/demo.ipynb) for demo visualization.

## Environment
* Python==3.6.12
* tensorflow==2.4.1
* tensorflow-gpu==2.2.0
* keras==2.4.3

More detail could be found in `requirements.txt`. (Some dependencies inside might not be necessary)

## Some Instructions
* `train.py`: training the baseline models. The trained models will be saved under `saved_models/` directory.
* `save_error.py`: getting and saving the errors or other information based on the trained models.
* `eval.py`: applying our method on the baselines.

Some example checkpoints (models or errors) are given via this [link](https://drive.google.com/file/d/1k7eMoMoVCg6tgJrTW8QHeFTtAGPO7_n2/view?usp=sharing). Unzip it and place the subdirectories under the repo.

## Citation
If you find this repo or our work useful for your research, please consider citing the paper
```
@inproceedings{deng2022cadet,
  title={CADET: Calibrated Anomaly Detection for Mitigating Hardness Bias},
  author={Deng, Ailin and Goodge, Adam and Ang, Lang Yi and Hooi, Bryan},
  booktitle={IJCAI},
  year={2022}
}
```
