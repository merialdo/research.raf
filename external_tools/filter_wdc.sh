#!/bin/bash
grep "\"keyValuePairs\":{" $1 | sed 's/$/,/g' | sed '$ s/,$/]/' | sed '1 s/^/[/' > $2
