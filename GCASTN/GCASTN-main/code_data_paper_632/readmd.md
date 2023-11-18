Our code runs in an environment built using docker

1. Building a docker runtime environment:

```
cd /code/docker
bash build.sh
```

2. generate missing data

   you can get complete data from [here](https://github.com/guoshnBJTU/ASTGNN/tree/main/data), and put them into `prepare_miss_data`,  set different parameters of function `Genaratepemsinputationdata` to get missing data of different types and different missing rates.

   get `RM`:

   ```
   python genarate_pems_data_for_imputation.py
   pyhton prepare_miss_data.py
   ```

   get `NM`:

   ```
   python genarate_pems_data_for_imputation.py
   pyhton prepare_miss_data.py
   python revise_NM_data.py
   ```

   get `BM`:

   ```
   python genarate_pems_data_for_imputation.py
   pyhton prepare_miss_data.py
   python revise_NM_data.py
   ```

3. run model

   put `true_data.npz` ,`PEMS04.csv`  and  `miss_data.npz` into `/code/GCASTN/data/PEMS04`

   then:

   ```
   python train_GCASTN.py
   ```

    