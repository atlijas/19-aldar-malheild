from collections import Counter
from re import finditer, sub
from itertools import combinations, product
from tokenizer import tokenize, correct_spaces
from transformers import pipeline

from .utility_functions import (gen_overlapping_ngrams,
                                get_differences,
                                get_token_case,
                                set_token_case,
                                merge_and_format)


from .lexicon_lookup import (exists_in_bin,
                            get_similar_from_tree,
                            BIN_TREE,
                            DICTIONARY,
                            DOUBLABLE_CONSONANTS,
                            all_parts_exist_in_bin,
                            JSON_EDITS)


ICEBERT_MODEL_PATH = 'transformer_models/IceBERT'

try:
    unmasker = pipeline(
        task='fill-mask',
        model=ICEBERT_MODEL_PATH,
        tokenizer=ICEBERT_MODEL_PATH,
        device=0
    )

except RuntimeError:
    unmasker = pipeline(
        task='fill-mask',
        model=ICEBERT_MODEL_PATH,
        tokenizer=ICEBERT_MODEL_PATH,
    )

MASK_TOK = unmasker.tokenizer.mask_token


YFIRLESTUR_MODEL = 'transformer_models/yfirlestur-icelandic-correction-byt5'
try:
    correct_ocr = pipeline(
                'text2text-generation',
                model=YFIRLESTUR_MODEL,
                tokenizer=YFIRLESTUR_MODEL,
                num_return_sequences=1,
                device=0
            )


except RuntimeError:
    correct_ocr = pipeline(
                'text2text-generation',
                model=YFIRLESTUR_MODEL,
                tokenizer=YFIRLESTUR_MODEL,
                num_return_sequences=1,
            )

# Returns all the combinations of possible edits, e.g.
# brezk-egypzku → [((3, 4), (10, 11)), ((3, 4),), ((10, 11),)]
def get_possible_edits_indices(token: str, orig: str) -> list:
    """
        Returns the indices of all the combinations of possible edits, e.g.
        brezk-egypzku → [((3, 4), (10, 11)), ((3, 4),), ((10, 11),)]
    """
    if len(orig) == 1:
        len_of_edits = len(orig) + 1
    else:
        len_of_edits = len(orig)
    orig_indices = [m.span() for m in finditer(orig, token)]
    edit_combs = list(combinations(orig_indices, len_of_edits)) + [(tuple(oi),) for oi in orig_indices]
    return edit_combs

def u_innskot(token: str) -> str:
    """
        Inserts 'u' into a token.
    """
    return token[:-1] + 'u' + token[-1]

def edit_ocurrence(token: str, replacement: str, indices: tuple) -> str:
    """
        Edits a single ocurrence, e.g. z→s, given a token and an index
    """
    return token[:indices[0]] + replacement + token[indices[1]:]

def edit_same_multiple_ocurrences(token: str, replacements: list, indices: tuple):
    """
        Edits all ocurrences of a single edits, e.g.
        brezk-egypsku → [bresk-egypsku, bresk-egypzku, brezk-egypsku]
    """
    token_to_return = token
    orig_len_diff = len(token) - len(token_to_return)
    curr_len_diff = orig_len_diff
    for i in indices:
        for replacement in replacements:
            curr_len_diff = len(token) - len(token_to_return)
            change_in_len = orig_len_diff - curr_len_diff
            token_to_return = edit_ocurrence(token_to_return, replacement, [w+change_in_len for w in i])
    return token_to_return

def double_consonant(token: str, consonant: str) -> list:
    """
       This function that accepts a token and a consonant, and creates 
       all the possible combinations of where the consonant could be 
       doubled and also where it has not been doubled,
       e.g. bygðarbygð → [byggðarbyggð, bygðarbyggð, byggðarbygð, bygðarbygð] and returns them all in a list.
    """
    indices = get_possible_edits_indices(token, consonant)
    possible_edits = []
    for index in indices:
        possible_edits.append(edit_same_multiple_ocurrences(token, [consonant*2], index))
    return possible_edits

def get_all_possible_modernized_versions(token: str) -> list:
    """
        Accepts a token and returns all possible single-n-gram-edit versions of it.
        A single-n-gram-edit means a single n-gram is edited in the token, once or more,
        e.g. islenzkur → [íslenzkur, islenskur...].
        TODO (maybe): Add a recursive step to edit the token further,
        e.g. islenzkur → íslenskur
    """
    uni_and_bigrams = list(gen_overlapping_ngrams(token, 1)) + list(gen_overlapping_ngrams(token, 2))
    possible_orig_chars = list({c for c in uni_and_bigrams if c in JSON_EDITS.keys()})
    # This could probably be made clearer:
    possible_edit_chars = [{oc: JSON_EDITS[oc]} for oc in possible_orig_chars]
    possible_edit_chars = {k:v for element in possible_edit_chars for k,v in element.items()}


    possible_edits = set()
    for pec in possible_edit_chars:
        possible_indices = list(get_possible_edits_indices(token, pec))
        for indices in possible_indices:
            for edit_char in possible_edit_chars[pec]:
                e_c = edit_char[0]
                current_orig_char_edits = edit_same_multiple_ocurrences(token, [e_c], indices)
                possible_edits.add(current_orig_char_edits)
    possible_edits = set(possible_edits)
    possible_edits.add(token)
    return list(possible_edits)


# FIXME: Tekur 100 ár. Kannski hægt að gera þetta á betri máta.
def modernize_parts(parts: list) -> list:
    """
        Accepts compound parts from Kvistur and tries to modernize their
        spelling. In its current form, this function is unusable, as it
        tries every single combination, which soon becomes too much.
    """
    modernized_parts = []
    for part in parts:
        modernized_parts.append(get_all_possible_modernized_versions(part))
    for i in product(*modernized_parts):
        yield ''.join(i)

def is_modernized(original_token, edited_token):
    """
        Accepts two tokens and returns a boolean value which represents
        whether edited_token could possibly be a modernized version of
        original_token, according to JSON_EDITS.
    """
    uni_and_bigrams = list(gen_overlapping_ngrams(original_token, 1)) + list(gen_overlapping_ngrams(original_token, 2))
    possible_orig_chars = list({c for c in uni_and_bigrams if c in JSON_EDITS.keys()})

    # Kannski óþarflega mikið rugl
    possible_edit_chars = [{oc: JSON_EDITS[oc]} for oc in possible_orig_chars]
    possible_edit_chars = {k:v for element in possible_edit_chars for k,v in element.items()}
    possible_edit_chars = {k: [v[0] for v in v] for k, v in possible_edit_chars.items() if v != []}
    differences = get_differences(original_token, edited_token)

    token_is_modernized = True


    for orig, edit in differences:
        if orig in possible_edit_chars:
            if edit not in possible_edit_chars[orig]:
                token_is_modernized = False
                break
        else:
            token_is_modernized = False
            break
    return token_is_modernized

def read_file(file):
    """
        Used to read a file and split it into lines.
    """
    with open(file, 'r', encoding='utf-8') as infile:
        #return sub('([^\dA-ZÁÉÍÓÚÝÞÆÖ]\.)', '\1\n', infile.read())
        return infile.read().replace('. ', '.\n').splitlines()

def get_best_from_mask(tokenized_sentence, cands, index):
    """
        Accepts a tokenized sentence and candidates that might replace the
        words @index, and tries them all @index. The tokens which the models
        decides are amongst the 100 most likely ones @index are returned.
    """
    # Búum til maskaða útgáfu af setningunni. Staðsetning maskans kemur frá indexinum.
    tmp_sent = ' '.join([i.txt for i in tokenized_sentence[:index]]) + f' {MASK_TOK} ' + ' '.join([i.txt for i in tokenized_sentence[index+1:]])
    # Fáum kandídata sem líkanið stingur upp á á stað maskatókans.
    unmasked_cands = unmasker(tmp_sent, top_k=100)
    # Fáum bara orðmyndir kandídatsins
    unmasked_cands = [w['token_str'].lstrip() for w in unmasked_cands]
    # Fáum sniðmengi kandídata úr líkaninu og kandídata úr args
    possible_cands_by_mask = [w for w in unmasked_cands if w in cands]
    # Ef engin orðmynd fæst notum við upphaflega tókann
    if len(possible_cands_by_mask) == 0:
        return None
    # Annars notum við orðmyndina sem líkanið gefur hæsta skorið í stað upphaflega tókans
    return possible_cands_by_mask[0]

def modernize_with_yfirlestur(token):
    """
        Accepts a single token and uses the yfirlestur model
        to correct it. The model is too greedy for whole sentences
        and edits them too much.
    """
    return correct_ocr(token)[0]['generated_text']

def modernize_sentence(sentence: str, check_similar_in_bin=True, check_parts_in_bin=True, check_modernized=True, check_yfirlestur=False, check_mask=False):
    """
        This function accepts a sentence and loops over its tokens and tries to
        modernize the ones not present in known lexicons.
    """
    # Tókum setninguna
    tokenized = list(tokenize(sentence))
    # Búum til tóman streng sem verður að endanlegu úttaki, þ.e. setningunni með nútímavæddri stafsetningu
    sentence_out = ''
    # Þurfum indexana (kannski) til að vita hvað við ætlum að maska
    for index, token in enumerate(tokenized):

        current_token = token.txt
        current_token_out = current_token
        # Höldum utan um há- og lágstafi, því við vinnum með lágstafað orð en þurfum að geta breytt því
        # til baka þegar við bætum því við málsgreinina.
        token_case = get_token_case(current_token)

        # Tékkar hvort tóki er orð (en ekki t.d. greinarmerki)
        # Ef hann er ekki orð er honum einfaldlega bætt við setninguna
        if token.kind == 6:

            # Það einfaldar málið að vinna með lágstafað orð
            current_token = current_token.lower()

            # Tékkum fyrst hvort orðið er til í BÍN. Ef svo er bætum við því við setninguna.
            # Þetta á væntanlega við um mörg (flest?) orð, þó textarnir séu gamlir.
            # Þetta gæti boðið hættunni heim. Ég veit það ekki.
            if exists_in_bin(current_token):
                current_token_out = current_token
                # TODO: Ef orð er bara til með hástaf í upphafi þarf að breyta token_case í 'title'

            # Ef orðið er ekki til í BÍN skoðum við hvort það sé til í leiðréttingarorðabókinni hans Finns
            # Í henni er 6281 þekkt vörpun <gamalt orð → nútímavætt orð> (3. maí 2023)
            # Þekkir algengustu breytingarnar, t.d. 'opt' → 'oft', 'vjer' → 'vér'
            # TODO: Stækka leiðréttingarorðabókina. Ég held að það gæti skorið af tímanum, sé litið fram á við.
            # En ef við keyrum þetta bara einu sinni á 19. aldar textunum er það kannski ekki svo mikilvægt.
            elif current_token in DICTIONARY:
                current_token_out = DICTIONARY[current_token]

            # Við reynum ekki einu sinni að breyta orðum sem eru styttri en 3 bókstafir.
            elif len(current_token) < 3:
                current_token_out = current_token

            # Ef orðið finnst hvorki í BÍN né í leiðréttingarorðabókinni
            # Þessi partur hér fyrir neðan er hægur. Það þyrfti helst að gera hann hraðari.
            else:
                plausible_candidates = []
                # Athugum hvort allir orðhlutar finnist í BÍN
                # Kvistur sér um að skipta orðinu í orðhluta
                if check_parts_in_bin:
                    if all_parts_exist_in_bin([current_token]):
                        # Ef allir orðhlutarnir eru til í BÍN bætum við upprunalega tókanum við
                        # mögulega kandídata
                        plausible_candidates.append(current_token)

                if check_similar_in_bin:
                    # Athugum við hvort svipuð orð (miðað við Levenshtein-fjarlægð) finnist í BÍN
                    # Skilar t.d. 'blandaðann' → ['blandaðan', 'blandarann']
                    # Leyfileg Levenshtein-fjarlægð er 2 fyrir tóka sem eru 12 stafir eða lengri, annars 1
                    # Hægt er að breyta þessu með breytunni 'lev_dist' í fallinu get_similar_from_tree
                    similar_from_bin = get_similar_from_tree(current_token, BIN_TREE)

                    # Skoða hvort þessi svipuðu orð séu nútímaafbrigði upprunalega orðsins
                    modernized_from_similar_in_bin = [token for token in similar_from_bin if is_modernized(current_token, token)]


                    # Bætum nútímavæddum orðum við listann yfir mögulega kandídata
                    plausible_candidates.extend(modernized_from_similar_in_bin)

                if check_modernized:
                    # Búum til lista af öllum mögulegum orðmyndum út frá þekktum skiptingum í edits.json
                    # Margar þeirra verða algert rugl. Tólið veit t.d. að d→ð er algeng skipting en það er auðvitað ekki alltaf rétt, t.d. í 'hundur' → 'hunður'
                    # Hér er einungis gert ráð fyrir breytingu á einum staf (þó á öllum stöðum í orðinu), t.d. 'bygðarbygð' → ['bygðarbyggð', 'byggðarbygð', 'byggðarbyggð']
                    # og 'islenzkur' → ['islenskur', 'íslenzkur']
                    
                    all_possible_modern = get_all_possible_modernized_versions(current_token)
                    # Hreinsum út þær orðmyndir sem eru ekki til í BÍN
                    all_possible_modern_in_bin = [i for i in all_possible_modern if exists_in_bin(i)]
                    plausible_candidates.extend(all_possible_modern_in_bin)

                # TODO (kannski): Rúlla aftur í gegnum tilbúnu orðmyndirnar og búa til nútímavædda útgáfu af þeim

                # Ef engin tilbúnu orðmyndanna er í BÍN skoðum við hvort einhver þeirra sé til með u-innskoti
                if current_token.endswith('r'):
                    with_u_innskot = u_innskot(current_token)
                    if exists_in_bin(with_u_innskot):
                        plausible_candidates.append(with_u_innskot)


                if any(i in current_token for i in DOUBLABLE_CONSONANTS):
                    for i in DOUBLABLE_CONSONANTS:
                        doubled_consonant = double_consonant(current_token, i)
                        plausible_candidates.extend([i for i in doubled_consonant if exists_in_bin(i)])

                else:
                    if check_yfirlestur:
                        best_from_yfirlestur = modernize_with_yfirlestur(current_token)
                        if best_from_yfirlestur and is_modernized(current_token, best_from_yfirlestur):
                            plausible_candidates.append(best_from_yfirlestur)
                    if check_mask:
                        # Við erum með try-klausu hérna því sumar setningar eru of langar fyrir líkanið.
                        # Það væri hægt að brjóta setninguna niður í n-gröm.
                        try:
                            best_from_masked = get_best_from_mask(tokenized, plausible_candidates, index)
                            if best_from_masked and is_modernized(current_token, best_from_masked):
                                plausible_candidates.append(best_from_masked)
                        except:
                            pass
                if plausible_candidates:
                    filtered = [i for i in plausible_candidates if is_modernized(current_token, i)]
                    most_common_suggestions = Counter(filtered).most_common()
                    if most_common_suggestions:
                        current_token_out = most_common_suggestions[0][0]
                else:
                    current_token_out = current_token
        else:
            current_token_out = current_token + ' '
        cased_token = set_token_case(current_token_out, token_case)
        sentence_out += f' {cased_token} '
    return correct_spaces(sentence_out)


if __name__ == '__main__':
    test_file = read_file('eval_data/gefn.corrected')
    merged = merge_and_format(test_file)
    for sent in merged:
        print(modernize_sentence(sent))
