for i in STRING_textmining; do
    echo $i;
    mysql -h knowcharles.dyndns.org -uroot -pKnowEnG --execute "use KnowNet; \
        select distinct n1_id, n2_id, weight, '$i' \
        from edge as e \
        join node_species as ns \
        where ns.node_id=e.n2_id \
        and ns.taxon=9606 \
        and e.et_name='$i';" \
        | tail -n +2 > KN_03.$i.edge;
done;

