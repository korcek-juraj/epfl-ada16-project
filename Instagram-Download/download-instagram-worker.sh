wget -q -O "$2.jpg" $3
hadoop fs -moveFromLocal "$2.jpg" "/user/fboehm/instagram-images/$1/$2.jpg"
