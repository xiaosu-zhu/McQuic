if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i "1 s|$| -O|" "$PREFIX"/bin/mcquic*
elif [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i "" "1 s|$| -O|" "$PREFIX"/bin/mcquic*
else
    sed -i "1 s|$| -O|" "$PREFIX"/bin/mcquic*
fi
