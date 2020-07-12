Trong note này em liệt kê các bước dùng tool blast để tính hiệu suất trên tập AMD.

- Update 12/07/2020

#### Các bước chuẩn bị dữ liệu cho AMD
##### Preprocess dữ liệu đem đi phân cụm
- Input: 2 file fasta.13696_environmental_sequence.020 và file fasta.13696_environmental_sequence.007.
- Output: nối file xxx.020 vào đuôi của file xxx.007 được 1 file chứa các read hoàn chỉnh.

##### Tạo database theo format của blast cho các tập chứa scaffold
- Nối các file fasta (mỗi file chứa scaffold của 1 loài) thành một file fasta duy nhất -> **input.fna**.
- Với file fasta chứa scaffold (**input.fna**) của các loài, tạo database cho nó theo format của blast như sau (**blastdb**), với **-dbtype nucl** là kiểu database nucleotide (vì mình sẽ dùng blastn để align từ nucleotide sang nucleotide):

```
    makeblastdb -in input.fna -parse_seqids -dbtype nucl -out blastdb
```

#### Kết quả phân cụm

- Parameters: k-mer, l-mer, number of shared reads, maximum seed size đều dùng giống của tập R (long read).

<img width="900" alt="Clustering result" src="img/pred-seed-latent-fasta.13696_environmental_sequence.png">

Mô tả kết quả phân cụm.

#### Quy trình dùng blast để align kết quả

- Từ kết quả phân cụm, em tạo ra 5 file fasta format tương ứng với 5 cluster, chứa các read thuộc về cụm đó, mỗi file có format như sau:

```
    \>seq0

    NNNNNNGGGATTGCTGGTCCCCTAAACTTACTAAGTGCAAATTAAAGAGGTTAATGGCCTAAGACAGTGTGGAGGTAGGCT

    \>seq1

    AGTGCATTAAAGAGGTTAATGGCCTAAGACAGTGTGGAGGTAGGCTTAGAAGCAGCCATCCTTCAAAGAGTGCGTAACAGC

    ...

```
- Sau đó, với mỗi file kết quả phân cụm, em dùng tool **blastn** (translate nucleotide -> nucleotide) như sau:

    * Em cũng đã thử dùng tool **blastx** như gợi ý (translate nucliotide sang protein) nhưng kết quả không tìm thấy (không có hits found nào) và chạy rất lâu (**blastp** cũng vậy).

```
    blastn.exe -db blastdb -query cluster_1.fna -out blast_result_1.txt
```

- Với **cluster_1.fna** chứa các read thuộc về cụm 1 (cluster_2.fna là cụm 2, ...). **blastdb** là database chứa scaffold của các loài (đã tạo ở bước trên).

- Mỗi cụm được query lần lượt với database -> 5 lần chạy tool blastn.

#### Cách đánh giá hiệu năng phân cụm

##### Kết quả sử dụng tool blast

- Với output từ tool blast, mỗi cụm sẽ có kết quả như sau:

- Gọi các loài có trong database như sau:

    * Thermoplasmatales archaeon Gpl Thermo_Gpl_Scaffold	:   specie 1
    * Ferroplasma acidarmanus Type I Ferro_acid_Scaffold	:	specie 2
    * Leptospirillum sp. Group III LeptoIII_Scaffold	:	    specie 3
    * Ferroplasma sp. Type II FerroII_Scaffold	:	            specie 4
    * Leptospirillum sp. Group II '5-wayCG' Scaffold	:	    specie 5

Bảng kết quả phân cụm, mỗi hàng tương úng với mỗi cụm được giải thuật trả về, mỗi cột tương ứng với kết quả từ blast.

|Cluster\Species   |1  |2  |3  |4  |5  |
|-----|---|---|---|---|---|
|**1**|66   |33   |10   |31   |35   |
|**2**|2024   |1374   |1667   |2485   |5936   |
|**3**|20957   | 11472   |790   | 29570   |4871   |
|**4**|1202   |595   |7399   |1106   |36811   |
|**5**|0   |0   |3   |0   |1   |

##### Tính hiệu năng

- Từ kết quả của tool blast, có thể gán cho mỗi cụm 1 groundtruth nhất định để tính precision, recall, f1 score.
- Có 5 loài và 5 cụm trả về thì có nhiều cách gán groundtruth cho cụm, ví dụ: cluster 1 - specie 1, cluster 1 - specie 2,... -> 5! cách
- Với mỗi cách gán, mình sẽ có giá trị f1, tính hết tất cả các giá trị f1 (5! giá trị) và chọn giá trị lớn nhất.

##### Kết quả
- Với 5 cluster: f1 = 35%
- Với 3 cluster (bỏ bớt cluster 1 và 5 do số điểm dữ liệu của 2 cụm này quá ít): f1 = 50%