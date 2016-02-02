import _mysql

db = _mysql.connect(host="knowcharles.dyndns.org", user="root", passwd="KnowEnG", db="KnowNet")

queryString = """SELECT distinct n1_id, n2_id, weight, {edge}
                 FROM edge AS e
                 JOIN node_species AS ns
                 WHERE ns.node_id=e.n2_id
                 AND ns.taxon=9606
                 AND e.et_name={edge}""".format(edge="kegg_pathway")

db.query(queryString)





