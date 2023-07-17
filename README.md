# Replication of paper: Cellular QoE Prediction for Video Service Based on Causal Structure Learning

This project is a code replication of "Cellular QoE Prediction for Video Service Based on Causal Structure Learning" as described in the published work. As the original paper did not release the used dataset, the replication has been tested on synthetically generated random data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you proceed, make sure you have installed the necessary packages and libraries. You should have Python>3.7 installed on your machine. The specific python libraries needed should be installed via pip:

```
pip install -r requirements.txt
```

### Data Generation

To generate synthetic KPI/KQI data, run the following command:

```
python gen.py
```

This will generate the data in the `./data/` directory. You should be able to find the generated dataset at `./data/cellular/cellular.csv`.

### Configuration

You can modify the network parameters and training hyperparameters in the `configs.py` file. Please consult this file and make changes as necessary for your specific application.

### Training and Testing

To start the training and testing process, run the following command:

```
python main.py
```

This will start the training process and test the model after training. 

## Outputs

The best-performing checkpoint from all epochs on the validation set, as well as the test performance of the model, will be saved in `best_model.pt`.

## Authors

* **Shaoyu Dou (窦绍瑜)** - Ph.D student, Tongji University, China.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Contact

For any questions or concerns, feel free to reach out to us. You can also raise an issue via GitHub's issue tracking system. If you need direct contact, please email the project maintainer at [shaoyu@tongji.edu.cn](mailto:shaoyu@tongji.edu.cn).

## Acknowledgments

* This project has been tested on synthetically generated data due to the unavailability of the original dataset.
* The following open-source projects have provided inspiration and code which this project builds upon:
  * [PyGAT](https://github.com/Diego999/pyGAT)
  * [transfer_entropy](https://github.com/notsebastiano/transfer_entropy)
