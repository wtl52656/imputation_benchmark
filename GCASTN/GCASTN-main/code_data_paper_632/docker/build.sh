docker build -t gcastn:last .

if [ $? -eq 0]; then
    docker tag gcastn:last a4-1:5000/huangyiheng/gcastn:last
    docker push a4-1:5000/huangyiheng/gcastn:last
fi
