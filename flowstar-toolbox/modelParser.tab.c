/* A Bison parser, made by GNU Bison 3.7.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.7"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 1 "modelParser.y"

/*---
  Email: Xin Chen <chenxin415@gmail.com> if you have questions or comments.
  The code is released as is under the GNU General Public License (GPL).
---*/


	#include "modelParser.h"

	extern int yyerror(const char *);
	extern int yyerror(std::string);
	extern int yylex();
	extern int yyparse();
	bool err;

	int lineNum = 1;

	std::vector<flowstar::Flowpipe> initialSets;

	void parseError(const char *str, int lnum)
	{
		std::cerr << "Error @line " << lineNum << ":" << std::string(str) << std::endl;
		exit(1);
	}

#line 97 "modelParser.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "modelParser.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_NUM = 3,                        /* NUM  */
  YYSYMBOL_IDENT = 4,                      /* IDENT  */
  YYSYMBOL_EXP = 5,                        /* EXP  */
  YYSYMBOL_SIN = 6,                        /* SIN  */
  YYSYMBOL_COS = 7,                        /* COS  */
  YYSYMBOL_LOG = 8,                        /* LOG  */
  YYSYMBOL_SQRT = 9,                       /* SQRT  */
  YYSYMBOL_UNIVARIATE_POLYNOMIAL = 10,     /* UNIVARIATE_POLYNOMIAL  */
  YYSYMBOL_MULTIVARIATE_POLYNOMIAL = 11,   /* MULTIVARIATE_POLYNOMIAL  */
  YYSYMBOL_EXPRESSION = 12,                /* EXPRESSION  */
  YYSYMBOL_GEQ = 13,                       /* GEQ  */
  YYSYMBOL_LEQ = 14,                       /* LEQ  */
  YYSYMBOL_EQ = 15,                        /* EQ  */
  YYSYMBOL_16_ = 16,                       /* '+'  */
  YYSYMBOL_17_ = 17,                       /* '-'  */
  YYSYMBOL_18_ = 18,                       /* '*'  */
  YYSYMBOL_19_ = 19,                       /* '/'  */
  YYSYMBOL_uminus = 20,                    /* uminus  */
  YYSYMBOL_21_ = 21,                       /* '^'  */
  YYSYMBOL_22_ = 22,                       /* '{'  */
  YYSYMBOL_23_ = 23,                       /* '}'  */
  YYSYMBOL_24_ = 24,                       /* '('  */
  YYSYMBOL_25_ = 25,                       /* ')'  */
  YYSYMBOL_26_ = 26,                       /* '['  */
  YYSYMBOL_27_ = 27,                       /* ','  */
  YYSYMBOL_28_ = 28,                       /* ']'  */
  YYSYMBOL_YYACCEPT = 29,                  /* $accept  */
  YYSYMBOL_input_content = 30,             /* input_content  */
  YYSYMBOL_multivariate_polynomial = 31,   /* multivariate_polynomial  */
  YYSYMBOL_univariate_polynomial = 32,     /* univariate_polynomial  */
  YYSYMBOL_expression = 33                 /* expression  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  8
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   142

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  29
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  5
/* YYNRULES -- Number of rules.  */
#define YYNRULES  36
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  92

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   271


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      24,    25,    18,    16,    27,    17,     2,    19,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    26,     2,    28,    21,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    22,     2,    23,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    20
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    63,    63,    69,    75,    83,    91,    99,   104,   112,
     129,   135,   154,   161,   171,   179,   187,   192,   200,   217,
     223,   239,   246,   254,   262,   267,   275,   281,   287,   294,
     300,   305,   313,   319,   325,   331,   337
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "NUM", "IDENT", "EXP",
  "SIN", "COS", "LOG", "SQRT", "UNIVARIATE_POLYNOMIAL",
  "MULTIVARIATE_POLYNOMIAL", "EXPRESSION", "GEQ", "LEQ", "EQ", "'+'",
  "'-'", "'*'", "'/'", "uminus", "'^'", "'{'", "'}'", "'('", "')'", "'['",
  "','", "']'", "$accept", "input_content", "multivariate_polynomial",
  "univariate_polynomial", "expression", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,    43,    45,    42,    47,
     271,    94,   123,   125,    40,    41,    91,    44,    93
};
#endif

#define YYPACT_NINF (-28)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int8 yypact[] =
{
      63,   -18,   -17,    25,    12,    42,    39,    31,   -28,   -28,
     -28,    42,    42,   111,   -28,   -28,    39,    39,    11,   119,
     -28,   -28,    29,    36,    38,    47,    57,    31,    31,    80,
     103,    64,    87,    42,    42,    42,    88,   -28,    72,    93,
      67,    39,    39,    39,    92,   -28,    31,    31,    31,    31,
      31,    86,    -8,    74,    31,    31,    31,    31,   110,   -28,
     -28,    23,    23,    64,   -28,   -28,   112,    43,    43,    72,
     -28,    33,    51,    61,    71,    81,   -28,   113,    -3,    -3,
      86,    86,   -28,    56,   -28,   -28,   -28,   -28,   -28,    89,
     -28,   -28
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     1,    21,
      20,     0,     0,     0,    13,    11,     0,     0,     0,     0,
      30,    28,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    19,     0,     0,     0,     0,     0,     3,    10,     0,
       0,     0,     0,     0,     0,     2,     0,     0,     0,     0,
       0,    27,     0,     0,     0,     0,     0,     0,     0,     4,
      16,    14,    15,    17,    18,     7,     0,     5,     6,     8,
       9,     0,     0,     0,     0,     0,    24,     0,    22,    23,
      25,    31,    26,     0,    32,    33,    34,    35,    36,     0,
      12,    29
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -28,   -28,   -10,    -9,   -27
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     4,    19,    13,    30
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
{
      51,    52,    31,    32,     5,     6,    38,    39,    54,    55,
      56,    57,     8,    58,    40,    56,    57,    76,    58,    71,
      72,    73,    74,    75,    61,    62,    63,    78,    79,    80,
      81,    67,    68,    69,    20,    21,    22,    23,    24,    25,
      26,    35,    14,    15,    36,     9,    10,     7,    27,    54,
      55,    56,    57,    46,    58,    28,    16,    29,    84,    11,
      47,    43,    48,    17,    44,    18,    12,    54,    55,    56,
      57,    49,    58,     1,     2,     3,    85,    54,    55,    56,
      57,    50,    58,    53,    90,    36,    86,    54,    55,    56,
      57,    64,    58,    44,    66,    70,    87,    54,    55,    56,
      57,    77,    58,    33,    34,    35,    88,    58,    36,    41,
      42,    43,    60,    82,    44,    83,    89,    91,    65,    54,
      55,    56,    57,     0,    58,     0,    59,    33,    34,    35,
       0,     0,    36,     0,    37,    41,    42,    43,     0,     0,
      44,     0,    45
};

static const yytype_int8 yycheck[] =
{
      27,    28,    11,    12,    22,    22,    16,    17,    16,    17,
      18,    19,     0,    21,     3,    18,    19,    25,    21,    46,
      47,    48,    49,    50,    33,    34,    35,    54,    55,    56,
      57,    41,    42,    43,     3,     4,     5,     6,     7,     8,
       9,    18,     3,     4,    21,     3,     4,    22,    17,    16,
      17,    18,    19,    24,    21,    24,    17,    26,    25,    17,
      24,    18,    24,    24,    21,    26,    24,    16,    17,    18,
      19,    24,    21,    10,    11,    12,    25,    16,    17,    18,
      19,    24,    21,     3,    28,    21,    25,    16,    17,    18,
      19,     3,    21,    21,    27,     3,    25,    16,    17,    18,
      19,    27,    21,    16,    17,    18,    25,    21,    21,    16,
      17,    18,    25,     3,    21,     3,     3,    28,    25,    16,
      17,    18,    19,    -1,    21,    -1,    23,    16,    17,    18,
      -1,    -1,    21,    -1,    23,    16,    17,    18,    -1,    -1,
      21,    -1,    23
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,    10,    11,    12,    30,    22,    22,    22,     0,     3,
       4,    17,    24,    32,     3,     4,    17,    24,    26,    31,
       3,     4,     5,     6,     7,     8,     9,    17,    24,    26,
      33,    32,    32,    16,    17,    18,    21,    23,    31,    31,
       3,    16,    17,    18,    21,    23,    24,    24,    24,    24,
      24,    33,    33,     3,    16,    17,    18,    19,    21,    23,
      25,    32,    32,    32,     3,    25,    27,    31,    31,    31,
       3,    33,    33,    33,    33,    33,    25,    27,    33,    33,
      33,    33,     3,     3,    25,    25,    25,    25,    25,     3,
      28,    28
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int8 yyr1[] =
{
       0,    29,    30,    30,    30,    31,    31,    31,    31,    31,
      31,    31,    31,    31,    32,    32,    32,    32,    32,    32,
      32,    32,    33,    33,    33,    33,    33,    33,    33,    33,
      33,    33,    33,    33,    33,    33,    33
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     4,     4,     4,     3,     3,     3,     3,     3,
       2,     1,     5,     1,     3,     3,     3,     3,     3,     2,
       1,     1,     3,     3,     3,     3,     3,     2,     1,     5,
       1,     3,     4,     4,     4,     4,     4
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
# ifndef YY_LOCATION_PRINT
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yykind < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yykind], *yyvaluep);
# endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* input_content: MULTIVARIATE_POLYNOMIAL '{' multivariate_polynomial '}'  */
#line 64 "modelParser.y"
{
	flowstar::multivariate_polynomial_setting.result = *(yyvsp[-1].intPoly);
	delete (yyvsp[-1].intPoly);
}
#line 1173 "modelParser.tab.c"
    break;

  case 3: /* input_content: UNIVARIATE_POLYNOMIAL '{' univariate_polynomial '}'  */
#line 70 "modelParser.y"
{
	flowstar::up_parseresult = *(yyvsp[-1].uniPoly);
	delete (yyvsp[-1].uniPoly);
}
#line 1182 "modelParser.tab.c"
    break;

  case 4: /* input_content: EXPRESSION '{' expression '}'  */
#line 76 "modelParser.y"
{
	flowstar::expression_setting.result = *(yyvsp[-1].pIntExpression);
	delete (yyvsp[-1].pIntExpression);
}
#line 1191 "modelParser.tab.c"
    break;

  case 5: /* multivariate_polynomial: multivariate_polynomial '+' multivariate_polynomial  */
#line 84 "modelParser.y"
{
	(yyval.intPoly) = (yyvsp[-2].intPoly);
	*(yyval.intPoly) += *(yyvsp[0].intPoly);

	delete (yyvsp[0].intPoly);
}
#line 1202 "modelParser.tab.c"
    break;

  case 6: /* multivariate_polynomial: multivariate_polynomial '-' multivariate_polynomial  */
#line 92 "modelParser.y"
{
	(yyval.intPoly) = (yyvsp[-2].intPoly);
	*(yyval.intPoly) -= *(yyvsp[0].intPoly);

	delete (yyvsp[0].intPoly);
}
#line 1213 "modelParser.tab.c"
    break;

  case 7: /* multivariate_polynomial: '(' multivariate_polynomial ')'  */
#line 100 "modelParser.y"
{
	(yyval.intPoly) = (yyvsp[-1].intPoly); 
}
#line 1221 "modelParser.tab.c"
    break;

  case 8: /* multivariate_polynomial: multivariate_polynomial '*' multivariate_polynomial  */
#line 105 "modelParser.y"
{
	(yyval.intPoly) = (yyvsp[-2].intPoly);
	*(yyval.intPoly) *= *(yyvsp[0].intPoly);

	delete (yyvsp[0].intPoly);
}
#line 1232 "modelParser.tab.c"
    break;

  case 9: /* multivariate_polynomial: multivariate_polynomial '^' NUM  */
#line 113 "modelParser.y"
{
	int exp = (int) (yyvsp[0].dblVal);

	if(exp == 0)
	{
		(yyval.intPoly) = new flowstar::Polynomial<flowstar::Interval>(1, flowstar::multivariate_polynomial_setting.pVars->size());
	}
	else
	{
		(yyval.intPoly) = new flowstar::Polynomial<flowstar::Interval>;
		(*(yyvsp[-2].intPoly)).pow(*(yyval.intPoly), exp);
	}

	delete (yyvsp[-2].intPoly);
}
#line 1252 "modelParser.tab.c"
    break;

  case 10: /* multivariate_polynomial: '-' multivariate_polynomial  */
#line 130 "modelParser.y"
{
	(yyval.intPoly) = (yyvsp[0].intPoly);
	*(yyval.intPoly) *= -1;
}
#line 1261 "modelParser.tab.c"
    break;

  case 11: /* multivariate_polynomial: IDENT  */
#line 136 "modelParser.y"
{
	int id = flowstar::multivariate_polynomial_setting.pVars->getIDForVar(*(yyvsp[0].identifier));

	if(id < 0)
	{
		char errMsg[MSG_SIZE];
		sprintf(errMsg, "Variable %s is not declared.", (*(yyvsp[0].identifier)).c_str());
		parseError(errMsg, lineNum);
		exit(1);
	}

	unsigned int numVars = flowstar::multivariate_polynomial_setting.pVars->size();
	(yyval.intPoly) = new flowstar::Polynomial<flowstar::Interval>(1, numVars);
	(yyval.intPoly)->mul_assign(id, 1);

	delete (yyvsp[0].identifier);
}
#line 1283 "modelParser.tab.c"
    break;

  case 12: /* multivariate_polynomial: '[' NUM ',' NUM ']'  */
#line 155 "modelParser.y"
{
	unsigned int numVars = flowstar::multivariate_polynomial_setting.pVars->size();
	flowstar::Interval I((yyvsp[-3].dblVal), (yyvsp[-1].dblVal));
	(yyval.intPoly) = new flowstar::Polynomial<flowstar::Interval>(I, numVars);
}
#line 1293 "modelParser.tab.c"
    break;

  case 13: /* multivariate_polynomial: NUM  */
#line 162 "modelParser.y"
{
	unsigned int numVars = flowstar::multivariate_polynomial_setting.pVars->size();
	(yyval.intPoly) = new flowstar::Polynomial<flowstar::Interval>((yyvsp[0].dblVal), numVars);
}
#line 1302 "modelParser.tab.c"
    break;

  case 14: /* univariate_polynomial: univariate_polynomial '+' univariate_polynomial  */
#line 172 "modelParser.y"
{
	(yyval.uniPoly) = (yyvsp[-2].uniPoly);
	(*(yyval.uniPoly)) += (*(yyvsp[0].uniPoly));

	delete (yyvsp[0].uniPoly);
}
#line 1313 "modelParser.tab.c"
    break;

  case 15: /* univariate_polynomial: univariate_polynomial '-' univariate_polynomial  */
#line 180 "modelParser.y"
{
	(yyval.uniPoly) = (yyvsp[-2].uniPoly);
	(*(yyval.uniPoly)) -= (*(yyvsp[0].uniPoly));

	delete (yyvsp[0].uniPoly);
}
#line 1324 "modelParser.tab.c"
    break;

  case 16: /* univariate_polynomial: '(' univariate_polynomial ')'  */
#line 188 "modelParser.y"
{
	(yyval.uniPoly) = (yyvsp[-1].uniPoly); 
}
#line 1332 "modelParser.tab.c"
    break;

  case 17: /* univariate_polynomial: univariate_polynomial '*' univariate_polynomial  */
#line 193 "modelParser.y"
{
	(yyval.uniPoly) = (yyvsp[-2].uniPoly);
	(*(yyval.uniPoly)) *= (*(yyvsp[0].uniPoly));

	delete (yyvsp[0].uniPoly);
}
#line 1343 "modelParser.tab.c"
    break;

  case 18: /* univariate_polynomial: univariate_polynomial '^' NUM  */
#line 201 "modelParser.y"
{
	int exp = (int) (yyvsp[0].dblVal);

	if(exp == 0)
	{
		(yyval.uniPoly) = new flowstar::UnivariatePolynomial<flowstar::Real>(1);
	}
	else
	{
		(yyval.uniPoly) = new flowstar::UnivariatePolynomial<flowstar::Real>;
		(*(yyvsp[-2].uniPoly)).pow(*(yyval.uniPoly), exp);
	}

	delete (yyvsp[-2].uniPoly);
}
#line 1363 "modelParser.tab.c"
    break;

  case 19: /* univariate_polynomial: '-' univariate_polynomial  */
#line 218 "modelParser.y"
{
	(yyval.uniPoly) = (yyvsp[0].uniPoly);
	(*(yyval.uniPoly)) *= -1;
}
#line 1372 "modelParser.tab.c"
    break;

  case 20: /* univariate_polynomial: IDENT  */
#line 224 "modelParser.y"
{
	std::string tVar("t");
	if((yyvsp[0].identifier)->compare(tVar) == 0)
	{
		(yyval.uniPoly) = new flowstar::UnivariatePolynomial<flowstar::Real>(1, 1);
	}
	else
	{
		char errMsg[MSG_SIZE];
		sprintf(errMsg, "The time variable should be denoted by t.");
		parseError(errMsg, lineNum);
		exit(1);
	}
}
#line 1391 "modelParser.tab.c"
    break;

  case 21: /* univariate_polynomial: NUM  */
#line 240 "modelParser.y"
{
	(yyval.uniPoly) = new flowstar::UnivariatePolynomial<flowstar::Real>((yyvsp[0].dblVal));
}
#line 1399 "modelParser.tab.c"
    break;

  case 22: /* expression: expression '+' expression  */
#line 247 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-2].pIntExpression);
	(*(yyval.pIntExpression)) += (*(yyvsp[0].pIntExpression));

	delete (yyvsp[0].pIntExpression);
}
#line 1410 "modelParser.tab.c"
    break;

  case 23: /* expression: expression '-' expression  */
#line 255 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-2].pIntExpression);
	(*(yyval.pIntExpression)) -= (*(yyvsp[0].pIntExpression));

	delete (yyvsp[0].pIntExpression);
}
#line 1421 "modelParser.tab.c"
    break;

  case 24: /* expression: '(' expression ')'  */
#line 263 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-1].pIntExpression); 
}
#line 1429 "modelParser.tab.c"
    break;

  case 25: /* expression: expression '*' expression  */
#line 268 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-2].pIntExpression);
	(*(yyval.pIntExpression)) *= (*(yyvsp[0].pIntExpression));

	delete (yyvsp[0].pIntExpression);
}
#line 1440 "modelParser.tab.c"
    break;

  case 26: /* expression: expression '^' NUM  */
#line 276 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-2].pIntExpression);
	(yyval.pIntExpression)->pow_assign((int)(yyvsp[0].dblVal));
}
#line 1449 "modelParser.tab.c"
    break;

  case 27: /* expression: '-' expression  */
#line 282 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[0].pIntExpression);
	(yyval.pIntExpression)->inv_assign();
}
#line 1458 "modelParser.tab.c"
    break;

  case 28: /* expression: IDENT  */
#line 288 "modelParser.y"
{
	(yyval.pIntExpression) = new flowstar::Expression<flowstar::Interval>(*(yyvsp[0].identifier), flowstar::expression_setting.pVars);

	delete (yyvsp[0].identifier);
}
#line 1468 "modelParser.tab.c"
    break;

  case 29: /* expression: '[' NUM ',' NUM ']'  */
#line 295 "modelParser.y"
{
	flowstar::Interval I((yyvsp[-3].dblVal), (yyvsp[-1].dblVal));
	(yyval.pIntExpression) = new flowstar::Expression<flowstar::Interval>(I);
}
#line 1477 "modelParser.tab.c"
    break;

  case 30: /* expression: NUM  */
#line 301 "modelParser.y"
{
	(yyval.pIntExpression) = new flowstar::Expression<flowstar::Interval>((yyvsp[0].dblVal));
}
#line 1485 "modelParser.tab.c"
    break;

  case 31: /* expression: expression '/' expression  */
#line 306 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-2].pIntExpression);
	(*(yyval.pIntExpression)) /= (*(yyvsp[0].pIntExpression));

	delete (yyvsp[0].pIntExpression);
}
#line 1496 "modelParser.tab.c"
    break;

  case 32: /* expression: EXP '(' expression ')'  */
#line 314 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-1].pIntExpression);
	(yyval.pIntExpression)->exp_assign();
}
#line 1505 "modelParser.tab.c"
    break;

  case 33: /* expression: SIN '(' expression ')'  */
#line 320 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-1].pIntExpression);
	(yyval.pIntExpression)->sin_assign();
}
#line 1514 "modelParser.tab.c"
    break;

  case 34: /* expression: COS '(' expression ')'  */
#line 326 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-1].pIntExpression);
	(yyval.pIntExpression)->cos_assign();
}
#line 1523 "modelParser.tab.c"
    break;

  case 35: /* expression: LOG '(' expression ')'  */
#line 332 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-1].pIntExpression);
	(yyval.pIntExpression)->log_assign();
}
#line 1532 "modelParser.tab.c"
    break;

  case 36: /* expression: SQRT '(' expression ')'  */
#line 338 "modelParser.y"
{
	(yyval.pIntExpression) = (yyvsp[-1].pIntExpression);
	(yyval.pIntExpression)->sqrt_assign();
}
#line 1541 "modelParser.tab.c"
    break;


#line 1545 "modelParser.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if !defined yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturn;
#endif


/*-------------------------------------------------------.
| yyreturn -- parsing is finished, clean up and return.  |
`-------------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 348 "modelParser.y"


int yyerror(const char * what)
{
	fprintf(stderr, "Error line %d: %s\n", lineNum, what);
	err = true;
	return 1;
}

int yyerror(std::string what)
{
	std::cerr << "Error line "<< lineNum << " " << what << std::endl;
	err = true;
	return 1;
}
