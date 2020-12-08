cd ~/meta-deep-binning
python metadec/dataset/data_pickling.py

zip -r pickle.zip data/hmp/hmp/pickle
mv pickle.zip ~/data-storage/hmp

cd ~/data-storage
git config --global user.email "hqb1595@gmail.com"
git config user.name "baohq"
git add hmp
git commit -m "latest pickle data"
git push origin master

