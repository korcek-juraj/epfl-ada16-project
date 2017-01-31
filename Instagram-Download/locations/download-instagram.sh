hadoop fs -mkdir "/user/fboehm/instagram-images/$MONTH"

hadoop fs -cat "/datasets/goodcitylife/$MONTH/*instagram*" |\
	../jq-linux64 -r '.[] | ._source.permalink' |\
	xargs -n2 -P10 ./download-instagram-worker.sh $MONTH
