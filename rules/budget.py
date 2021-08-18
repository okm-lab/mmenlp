DIGIT = INT.interpretation(
    interp.custom(int)
)

# # # #

LITERALS = {
    'двести': 200,
    'сто': 100,
    'девяносто': 90,
    'восемьдесят': 80,
    'шестьдесят': 60,
    'семьдесят': 70,
    'шестьдесят': 60,
    'пятьдесят': 50,
    'сорок': 40,
    'тридцать': 30,
    'двадцать': 20,
}

LITERAL = dictionary(LITERALS).interpretation(
    interp.normalized().custom(LITERALS.get)
)


# # # #

Minimal = fact('Minimal', [])


MINIMAL = morph_pipeline([
    'маленький',
    'скромный',
    'небольшой',
    'минимальный',
    'немного'
]).interpretation(
    Minimal
)


# # # #

Range = fact(
    'Range',
    ['start', 'stop']
)


VALUE = or_(
    DIGIT,
    LITERAL
)


RANGE_MINUS = rule(
    VALUE.interpretation(Range.start),
    '-',
    VALUE.interpretation(Range.stop)
)

RANGE_WORD = morph_pipeline([
    'около',
    'до'
])

RANGE_TEXT_UP = rule(
    RANGE_WORD,
    VALUE.interpretation(Range.stop)
)

RANGE_TEXT = rule(
    VALUE.interpretation(Range.start),
    RANGE_WORD,
    VALUE.interpretation(Range.stop)
)

RANGE = or_(
    RANGE_MINUS,
    RANGE_TEXT,
    RANGE_TEXT_UP
).interpretation(
    Range
)


# # # #

Amount = fact(
    'Amount',
    ['value']
)


AMOUNT = or_(
    VALUE,
    RANGE,
    MINIMAL
).interpretation(
    Amount
)

# # # #

Budget = fact(
    'Budget',
    ['budget_word', 'amount']
)

NAME = morph_pipeline([
    'бюджет',
    'затрата',
    'стоимость',
    'тысяч рублей',
    'рублей',
    'потратить'
])

B1 = rule(
    AMOUNT.interpretation(
        Budget.amount
    ),
    NAME.interpretation(
        Budget.budget_word
    )
)

B2 = rule(
	NAME.interpretation(
        	Budget.budget_word
	),
	AMOUNT.interpretation(
		Budget.amount
	)
)

BUDGET = or_(
    B1,
    B2
).interpretation(
    Budget
)
