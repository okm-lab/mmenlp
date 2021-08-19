from yargy.pipelines import morph_pipeline


Childrens = fact('Childrens', [])
Club = fact('Club', [])
Bed = fact('Bed', [])

CHILDRENS = morph_pipeline([
    'детский',
    'ребёнка',
    'детей',
    'младший',
    'старший',
    'немного',
    'детки',
    'младенец'
]).interpretation(
    Childrens
)

FOR = morph_pipeline([
    'для'
])

CLUB = morph_pipeline([
    'зона',
   	'клуб',
   	'кружок'
]).interpretation(
    Club
)

BED = morph_pipeline([
    'кроватка',
   	'кровать',
   	'колыбель'
]).interpretation(
    Bed
)

NECES = or_(
    BED,
    CLUB
).interpretation(
    Children_neces.neces
)

Children_neces = fact('Children_neces', ['childrens_word','neces'])

CHILDREN_NECES_1 = rule(
    NECES,
    FOR.optional(),
    CHILDRENS.interpretation(
        Children_neces.childrens_word
    )
)

CHILDREN_NECES_2 = rule(
    FOR.optional(),
    CHILDRENS.interpretation(
        Children_neces.childrens_word
    ),
    NECES
)

CHILDREN_NECES = or_(
    CHILDREN_NECES_1,
    CHILDREN_NECES_2
).interpretation(
    Children_neces
)

# # # #

WITHOUT = morph_pipeline([
    'без',
    'нет'
])

WO_CHILDREN = or_(
    rule(
        WITHOUT,
        CHILDRENS
    ),
    rule(
        CHILDRENS,
        WITHOUT
    )
)

# # # #

KidsNumber = fact('KidsNumber', ['words', 'amount'])

LITTLE = morph_pipeline([
    'маленький',
    'младший'
])

LITERAL = morph_pipeline([
    'два',
    'один',
    'три',
    'их',
    'её'
])

DIGIT = INT.interpretation(
    interp.custom(int)
)

AMOUNT = or_(
    LITTLE,
    DIGIT,
    LITERAL
).interpretation(
    KidsNumber.amount
)

KIDS_NUMBER_1 = rule(
    CHILDRENS.interpretation(
        KidsNumber.words
    ),
    AMOUNT
)

KIDS_NUMBER_2 = rule(
    AMOUNT,
    CHILDRENS.interpretation(
        KidsNumber.words
    )
)

KIDS_NUMBER = or_(
    WO_CHILDREN,
    morph_pipeline([
        'младенец'
    ]).interpretation(
        KidsNumber.words
    ),
    KIDS_NUMBER_1,
    KIDS_NUMBER_2
).interpretation(
    KidsNumber
)