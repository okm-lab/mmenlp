Minimal = fact('Minimal', [])
Childrens = fact('Childrens', [])
Club = fact('Club', [])
Bed = fact('Bed', [])

CHILDRENS = morph_pipeline([
    'детский',
    'ребёнка',
    'детей',
    'младший',
    'старший',
    'немного'
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
)

CHILDREN_NECES_1 = rule(
    NECES,
    FOR.optional(),
    CHILDRENS
)

CHILDREN_NECES_2 = rule(
    FOR.optional(),
    CHILDRENS,
    NECES
)

CHILDREN_NECES = or_(
    CHILDREN_NECES_1,
    CHILDREN_NECES_2
)
