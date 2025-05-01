import collections

def getPairStats(splits):
    pairCounts = collections.defaultdict(int)
    for wordTuple,freq in splits.items():
        symbols = list(wordTuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i],symbols[i+1])
            pairCounts[pair] += freq

    print(f"\nPair Counts are: {len(pairCounts)}")
    return pairCounts


def mergePair(pairToMerge,splits):
    newSplits = {}
    (first,second) = pairToMerge
    mergedToken = first + second

    for wordTuple,freq in splits.items():
        symbols = list(wordTuple)
        newSymbols = []
        i = 0

        while i < len(symbols):

            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                newSymbols.append(mergedToken)
                i+=2
            else:
                newSymbols.append(symbols[i])
                i+=1

        newSplits[tuple(newSymbols)] = freq

    return newSplits


def main():
    corpus = ["He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist.",
              "He is often called England's national poet and the Bard of Avon or simply 'the Bard.",
              "His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems and a few other verses, some of uncertain authorship.",
              "His plays have been translated into every major living language and are performed more often than those of any other playwright.",
              "Shakespeare remains arguably the most influential writer in the English language, and his works continue to be studied and reinterpreted.",
              "Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire.",
              "At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith.",
              "Sometime between 1585 and 1592, he began a successful career in London as an actor, writer, and part-owner (sharer) of a playing company called the Lord Chamberlain's Men, later known as the King's Men after the ascension of King James VI of Scotland to the English throne.",
              "At age 49 (around 1613), he appears to have retired to Stratford, where he died three years later.",
              "Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and even certain fringe theories as to whether the works attributed to him were written by others.",
              "Shakespeare produced most of his known works between 1589 and 1613.",
              "His early plays were primarily comedies and histories and are regarded as some of the best works produced in these genres.",
              "He then wrote mainly tragedies until 1608, among them Hamlet, Othello, King Lear and Macbeth, all considered to be among the finest works in English.",
              "In the last phase of his life, he wrote tragicomedies (also known as romances) such as The Winter's Tale and The Tempest, and collaborated with other playwrights.",
              "Many of Shakespeare's plays were published in editions of varying quality and accuracy during his lifetime.",
              "However, in 1623, John Heminges and Henry Condell, two fellow actors and friends of Shakespeare's, published a more definitive text known as the First Folio, a posthumous collected edition of Shakespeare's dramatic works that includes 36 of his plays.",
              "Its Preface includes a prescient poem by Ben Jonson, a former rival of Shakespeare, who hailed Shakespeare with the now famous epithet: not of an age, but for all time",
              "Shakespeare was the son of John Shakespeare, an alderman and a successful glover (glove-maker) originally from Snitterfield in Warwickshire, and Mary Arden, the daughter of an affluent landowning family.",
              "He was born in Stratford-upon-Avon, where he was baptised on 26 April 1564",
              "His date of birth is unknown but is traditionally observed on 23 April, Saint George's Day",
              "This date, which can be traced to William Oldys and George Steevens, has proved appealing to biographers because Shakespeare died on the same date in 1616",
              "He was the third of eight children, and the eldest surviving son",
              "Although no attendance records for the period survive, most biographers agree that Shakespeare was probably educated at the King's New School in Stratford, a free school chartered in 1553, about a quarter-mile (400 m) from his home",
              "Grammar schools varied in quality during the Elizabethan era, but grammar school curricula were largely similar: the basic Latin text was standardised by royal decree, and the school would have provided an intensive education in grammar based upon Latin classical authors.",
              "At the age of 18, Shakespeare married 26-year-old Anne Hathaway.",
              "The consistory court of the Diocese of Worcester issued a marriage licence on 27 November 1582",
              "The next day, two of Hathaway's neighbours posted bonds guaranteeing that no lawful claims impeded the marriage.",
              "The ceremony may have been arranged in some haste since the Worcester chancellor allowed the marriage banns to be read once instead of the usual three times, and six months after the marriage Anne gave birth to a daughter, Susanna, baptised 26 May 1583.",
              "Twins, son Hamnet and daughter Judith, followed almost two years later and were baptised 2 February 1585.",
              "Hamnet died of unknown causes at the age of 11 and was buried 11 August 1596.",
              "After the birth of the twins, Shakespeare left few historical traces until he is mentioned as part of the London theatre scene in 1592.",
              "The exception is the appearance of his name in the complaints bill of a law case before the Queen's Bench court at Westminster dated Michaelmas Term 1588 and 9 October 1589.",
              "Scholars refer to the years between 1585 and 1592 as Shakespeare's lost years.",
              "Biographers attempting to account for this period have reported many apocryphal stories.",
              "Nicholas Rowe, Shakespeare's first biographer, recounted a Stratford legend that Shakespeare fled the town for London to escape prosecution for deer poaching in the estate of local squire Thomas Lucy.",
              "Shakespeare is also supposed to have taken his revenge on Lucy by writing a scurrilous ballad about him.",
              "Another 18th-century story has Shakespeare starting his theatrical career minding the horses of theatre patrons in London.",
              "John Aubrey reported that Shakespeare had been a country schoolmaster.",
              "Some 20th-century scholars suggested that Shakespeare may have been employed as a schoolmaster by Alexander Hoghton of Lancashire, a Catholic landowner who named a certain William Shakeshafte in his will.",
              "Little evidence substantiates such stories other than hearsay collected after his death, and Shakeshafte was a common name in the Lancashire area."]


    print("Training corupus:")
    for doc in corpus:
        print(doc)

    #Now we are creating our vocabulary.
    uniqueChars = set()
    for doc in corpus:
        for char in doc:
            uniqueChars.add(char)

    vocab = list(uniqueChars)
    vocab.sort()
    print(f"\nVocab Size:{len(vocab)}\n",vocab[:25])

    #Now we are adding a special token called, "end of word"
    endOfWord = "</w>"
    vocab.append(endOfWord)

    wordSplits = {}
    for doc in corpus:
        words = doc.strip().split(' ')
        for word in words:
            if word:
                charList = list(word) + [endOfWord]
                wordTuple = tuple(charList)

                if wordTuple not in wordSplits:
                    wordSplits[wordTuple]  = 0

                wordSplits[wordTuple] += 1


    print("\nPre tokenized word frequencies")
    print(f"Count:{len(wordSplits)}\n",wordSplits)

    #----------

    numMerges = 20
    merges = {}
    currentSplits = wordSplits.copy()

    print("\nStarting BPE Merges")
    print(f"Initial Splits:{currentSplits}")
    print("-" * 30)

    for i in range(numMerges):
        print(f"\n Merge Iteration {i+1} /{numMerges}")
        pairStats = getPairStats(currentSplits)
        if not pairStats:
            print("No more pairs to merge")
            break

        sortedPairs = sorted(pairStats.items(),key=lambda item:item[1],reverse=True)
        print(f"\nTop 5 Pair Frequencies: {sortedPairs[:5]}")

        #Find best pair.
        bestPair = max(pairStats,key=pairStats.get)
        bestFreq = pairStats[bestPair]
        print(f"\nFound Best Pair: {bestPair} with frequency: {bestFreq}")

        currentSplits = mergePair(bestPair,currentSplits)
        newToken = bestPair[0] + bestPair[1]
        print(f"Merging {bestPair} into '{newToken}'")
        print(f"Splits after merge: {currentSplits}")

        vocab.append(newToken)
        print(f"Updated Vocabulary V: {vocab}")

        merges[bestPair] = newToken
        print(f"Updated Merges: {merges}")
        print("-" * 30)


    print("\nBPE Merges Complete ")

    print(f"Final Vocabulary Size: {len(vocab)}")
    print("\nLearned Merges (Pair --> New Token):")
    for pair,token in merges.items():
        print(f"{pair} --> {token}")

    print("\n Final Word Splits after all merges: ")
    print(currentSplits)

    print("\n Final Vocabulary V sorted:")
    finalVocabSorted = sorted(list(set(vocab)))
    print(finalVocabSorted)





if __name__ == '__main__':
    main()
#


