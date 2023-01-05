#! /bin/zsh
cnt=1
while true
do
    ./test.sh > stdout.log 2> stderr.log
    if [[ $? -ne 0 ]]; then
	break
    fi
    let "cnt++"
done

echo "$cnt"
cat stdout.log
cat stderr.log
