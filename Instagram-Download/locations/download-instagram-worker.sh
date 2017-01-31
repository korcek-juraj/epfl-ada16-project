INSTA_ID_SLASH=${2/https:\/\/instagram.com\/p\//}
FILE_NAME=${INSTA_ID_SLASH//\//}.json

curl -L "$2?__a=1" | ../jq-linux64 '.media.location | objects' > $FILE_NAME
[ -s $FILE_NAME ] && hadoop fs -moveFromLocal $FILE_NAME "/user/fboehm/instagram-images/$1/$FILE_NAME"
rm -- $FILE_NAME
