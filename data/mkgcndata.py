import argparse

def main(args):

    entity_ids = {}
    with open(args.entdatapath) as df:
        lines = df.readlines()
        for line in lines:
            name, id = line.strip().split('   ')
            entity_ids[name] = int(id)
    with open(args.adjrelpath) as f:
            lines = f.readlines()
            with open(args.interactionsfile, 'a') as wf:
                    wf.write('Weight,Src,Dst\n')
            for line in lines:
                c1, rel, c2 = line.strip().split('\t')
                #print('C1:{} Rel:{} C2:{}'.format(c1,rel,c2))
                c1_id = entity_ids[c1]
                c2_id = entity_ids[c2]
                with open(args.interactionsfile, 'a') as wf:
                    wf.write('{},{},{}\n'.format(int(1),c1_id,c2_id))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--entdatapath', dest='entdatapath', default='wikidata/kg_training_entids_withshapes.txt', help='Path to the file containing the entities and entity IDs')
    parser.add_argument('--adjrelpath', dest='adjrelpath', default='wikidata/adjacency_rels_countries.txt', help='Path to the file containing adjacency relations of KG nodes')
    parser.add_argument('--interactionsfile', dest='interactionsfile', default='wikidata/gcndata/interactions.csv', help='Path to the file containing adjacency relations of KG nodes')
    
    args = parser.parse_args()

    main(args)