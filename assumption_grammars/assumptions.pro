/**

  ---- Standard DCG based Assummption Grammars ----
  
  attempt to implement at source level some of the functionality
  of assumption grammars - by overloading the plain DCG transform
  
  - this basically provides AG functionality in any Prolog having DCGs
  
  OO + DCGs seem an extremely powerful mechanism
  
  NOTE: predicates defined here with arity 3 should be used within clause
        having DCG arrows
        
  porting an AG program to this is quite easy:
  
  replace :- by -->
  
  replace #X by '#:'(X)
  
  use phrase/3 to test out AG components without DCG clause - like in:
 

 ?-phrase(('#<'([a,b,c]),'#+'(t(99)),'#*'(p(88)),'#-'(t(A)),'#-'(p(B)),'#:'(X),'#>'(As)),Xs,Ys).
 
 which returns:
 
 A = 99 Ys = (['*'(p(88))|_408] / _408) - [b,c] Xs = _183 As = [b,c] X = a B = 88 
  
*/

/*
% not useful in xvm
:-op(300,xfx,'#<').
:-op(300,xfx,'#+').
:-op(300,xfx,'#*').
:-op(300,xfx,'#-').
:-op(300,xfx,'#:').
:-op(300,xfx,'#>').
*/

%% to test assumption grammars: ?-phrase(('#<'([a,b,c]),'#+'(t(99)),'#*'(p(88)),'#-'(t(A)),'#-'(p(B)),'#:'(X),'#>'(As)),Xs,Ys).

%% #<'(Xs): sets the dcg token list to be Xs for processing by assumption grammar
'#<'(Xs,_,Db-Xs):-new_assumption_db(Db).

%% '#>'(Xs): unifies current assumption grammar token list with Xs
'#>'(Xs,Db-Xs,Db-Xs).

%% '#:'(X): matches X against current dcg token the assumption grammar is working on
'#:'(X,Db-[X|Xs],Db-Xs).

%% '#+'(X): adds 'linear' assumption +(X) to be consumed at most once, by a '#-' operation 
'#+'(X,Db1-Xs,Db2-Xs):-add_assumption('+'(X),Db1,Db2).

%% '#*'(X): adds 'intuitionisic' assumption *(X) to be used indefinitely by '#-' operation
'#*'(X,Db1-Xs,Db2-Xs):-add_assumption('*'(X),Db1,Db2).

%% '#='(X): unifies X with any matching existing or future +(X) linear assumptions
'#='(X,Db1-Xs,Db2-Xs):-equate_assumption('+'(X),Db1,Db2).

%% '#-'(X): consumes +(X) linear assumption or matches *(X) intuitionistic assumption
'#-'(X,Db1-Xs,Db2-Xs):-consume_assumption('+'(X),Db1,Db2).
'#-'(X,Db-Xs,Db-Xs):-match_assumption('*'(X),Db).

%% '#?'(X): matches +(X) or *(X) assumptions without any binding
'#?'(X,Db-Xs,Db-Xs):-match_assumption('+'(X),Db).
'#?'(X,Db-Xs,Db-Xs):-match_assumption('*'(X),Db).

new_assumption_db(Xs/Xs).

add_assumption(X,Xs/[X|Ys],Xs/Ys).

consume_assumption(X,Xs/Ys,Zs/Ys):-nonvar_select(X,Xs,Zs).

match_assumption(X,Xs/_):-nonvar_member(X0,Xs),copy_term(X0,X).

equate_assumption(X,Xs/Ys,XsZs):- \+(nonvar_member(X,Xs)),!,add_assumption(X,Xs/Ys,XsZs).
equate_assumption(X,Xs/Ys,Xs/Ys):-nonvar_member(X,Xs).

% nonvar_member(X,XXs):-println(entering=nonvar_member(X,XXs)),fail.
nonvar_member(X,XXs):-nonvar(XXs),XXs=[X|_].
nonvar_member(X,YXs):-nonvar(YXs),YXs=[_|Xs],nonvar_member(X,Xs).

nonvar_select(X,XXs,Xs):-nonvar(XXs),XXs=[X|Xs].
nonvar_select(X,YXs,[Y|Ys]):-nonvar(YXs),YXs=[Y|Xs],nonvar_select(X,Xs,Ys).


/* backtrackable instance fields
dcg_set(Xs):- dcg<==Xs.

dcg_advance(X):-dcg==>[X|Xs],dcg<==Xs.

dcg_get(Xs):-dcg==>Xs.
*/

xvm_bug:-phrase(('#<'([a,b,c]),'#+'(t(99)),'#*'(p(88)),'#-'(t(A)),'#-'(p(B)),'#:'(X),'#>'(As)),Xs,Ys),write(X+A+B+As+Xs+Ys),nl.


sent([A,B,X,As]) -->
  #<([a,b,c]),
  #+(t(99)),
  #*(p(88)),
  #-(t(A)),
  #-(p(B)),
  #:(X),
  #>(As).

% ok in xvm with --> notation
go:-sent(R,_,_),write(R),nl,fail.


