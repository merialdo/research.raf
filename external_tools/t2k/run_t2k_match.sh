#!/bin/bash

JAR="t2kmatch-2.1-jar-with-dependencies.jar"
CLS="de.uni_mannheim.informatik.dws.t2k.match.T2KMatch"

# required parameters: Knowledge Base, Web Tables and Ontology (class hierarchy)
KB="dbpedia/"
ONT="OntologyDBpedia"
WEB="webtables/"

# optional parameters: Index location (will be created if it does not exist), Surface Forms and Redirects
IDX="temp/index/"
SF="data/surfaceforms.txt"
RD="data/redirects.txt"

# evaluation parameters (optional): gold standards for class, property and instance mapping
GS_CLS="data/gs_class.csv"
GS_PROP="data/gs_property.csv"
GS_INST="data/gs_instance.csv"

# output location
RES="temp/output/"

mkdir $RES

java -Xmx7G -cp $JAR $CLS -sf $SF -kb $KB -ontology $ONT -web $WEB -index $IDX -classGS $GS_CLS -identityGS $GS_INST -schemaGS $GS_PROP -results $RES -verbose