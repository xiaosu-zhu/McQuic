if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i "1 s|$| -OO|" "$PREFIX"/bin/mcquic*
elif [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i "" "1 s|$| -OO|" "$PREFIX"/bin/mcquic*
else
    sed -i "1 s|$| -OO|" "$PREFIX"/bin/mcquic*
fi
