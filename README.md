# meta-deep-binning

#### Setups
- Python: 3.6
```
pip install -r requirements.txt
```

#### Running

```
python main.py --data_dir data --dataset_name L1 --result_dir results
```

- **data_dir** contains raw as subdir that contains fasta files.
- After finish running, results will be save in **result_dir**/log and **result_dir**/model.
- If data files are placed as below tree, running is simple: **python main.py**

```
.
├── README.md
├── config
│   └── dataset_metadata.json
├── data
│   ├── processed
│   └── raw
│       ├── L1.fna
│       ├── L2.fna
│       ├── L3.fna
├── main.py
```