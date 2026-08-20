#!/bin/sh

# Target output directory
TARGET_DIR="/home/thor/Programming/IRSTD/methods/datasets/DAUB"

# Create the directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# List of URLs to download
URLS="
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b690&path=/V1/%E8%AF%84%E5%88%86%E7%A8%8B%E5%BA%8Fpython%E7%89%88.zip&fileName=%E8%AF%84%E5%88%86%E7%A8%8B%E5%BA%8Fpython%E7%89%88.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b691&path=/V1/data_label.rar&fileName=data_label.rar
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b692&path=/V1/data22.zip&fileName=data22.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b693&path=/V1/data7.zip&fileName=data7.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b694&path=/V1/data3.zip&fileName=data3.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b695&path=/V1/data13.zip&fileName=data13.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b696&path=/V1/data11.zip&fileName=data11.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b697&path=/V1/data18.zip&fileName=data18.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b698&path=/V1/data16.zip&fileName=data16.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b699&path=/V1/data2.zip&fileName=data2.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b69a&path=/V1/data21.zip&fileName=data21.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b69b&path=/V1/data20.zip&fileName=data20.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b69c&path=/V1/data1.zip&fileName=data1.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b69d&path=/V1/data4.zip&fileName=data4.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b69e&path=/V1/data19.zip&fileName=data19.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b69f&path=/V1/data15.zip&fileName=data15.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a0&path=/V1/data12.zip&fileName=data12.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a1&path=/V1/data8.zip&fileName=data8.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a2&path=/V1/data6.zip&fileName=data6.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a3&path=/V1/data_label.zip&fileName=data_label.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a4&path=/V1/data10.zip&fileName=data10.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a5&path=/V1/data5.zip&fileName=data5.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a6&path=/V1/data14.zip&fileName=data14.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a7&path=/V1/data9.zip&fileName=data9.zip
https://download.scidb.cn/download?fileId=5f5198d05e972d531d33b6a8&path=/V1/data17.zip&fileName=data17.zip
"

echo "Starting parallel downloads..."

for url in $URLS; do
    # Extract filename from the URL query string using sed (e.g., extracts data22.zip)
    filename=$(echo "$url" | sed -n 's/.*fileName=\([^&]*\).*/\1/p')
    
    # URL-decode the first filename if necessary (评分程序python版.zip)
    if [ "$filename" = "%E8%AF%84%E5%88%86%E7%A8%8B%E5%BA%8Fpython%E7%89%88.zip" ]; then
        filename="评分程序python版.zip"
    fi

    # Skip empty lines if any
    [ -z "$url" ] && continue

    echo "Downloading: $filename"
    
    # Run curl in the background using '&' 
    # -L follows redirects, -s hides the noisy progress meter
    curl -L -s -o "$TARGET_DIR/$filename" "$url" &
done

# Wait for all background curl processes to complete before exiting
wait
echo "All downloads completed!"