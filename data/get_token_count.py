import nltk
from nltk import word_tokenize

total_tokens_dialogue = 0
num_turns = 0
with open('perfect_segmented_targets.tsv', 'r') as f:
        lines = f.readlines()
        print("--------------------------------------------------------------------------")
        for line in lines:
            print('Utterances ====> {}'.format(line))
            game_id, country_name, utts = line.strip().split('\t')
            utterances =  utts.split('<br>')
            for utt in utterances:
                num_turns += 1
                tokens = nltk.word_tokenize(utt)
                print('Utterances:{} tokens:{}'.format(utt, tokens))
                total_tokens_dialogue += len(tokens)
print('Total number of turns in the data:{}'.format(num_turns))
print('Total number of tokens in the data:{}'.format(total_tokens_dialogue))