#!/bin/bash

set -e

mkdir -p ~/.kaggle

cp /content/drive/MyDrive/ColabNotebooks/kaggle_API_credentials/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c walmart-recruiting-store-sales-forecasting

unzip walmart-recruiting-store-sales-forecasting.zip
rm walmart-recruiting-store-sales-forecasting.zip

for file in *.zip; do
    unzip "$file"
    rm "$file"
done

echo "Data downloaded and extracted successfully."
