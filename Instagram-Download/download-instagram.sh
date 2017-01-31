hadoop fs -mkdir "/user/fboehm/instagram-images/$MONTH"

hadoop fs -ls "/user/fboehm/instagram-images/$MONTH/*.json" | cut -d/ -f6 | cut -d. -f1 |\
	./jq-linux64 -r '.[] | ((._source.permalink | sub("^https://instagram.com/p/"; "") | sub("/$"; "")) + " " + ._source.image_src)' |\
	xargs -n2 -P10 ./download-instagram-worker.sh $MONTH
