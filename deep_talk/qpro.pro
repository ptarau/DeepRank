:- dynamic(old_answer/1).

trace(0).

load_plugins:-  
  PF='plugins.pro',
  (exists_file(PF)->consult(PF);nb_setval(plugins,false)).
 
:-load_plugins.
 
has_plugins:-nb_getval(plugins,true).


% loads a .txt file - assumes Python has saved dep, sent and rank facts

% load the Python-generated Prolog fact for a given text file
load(FNameNoSuf):-is_loaded(FNameNoSuf),!.
load(FNameNoSuf):-
  atomic_list_concat([FNameNoSuf,'.pro'],F),
  do((
    member(FN,
     [sim/4,sent/2,ner/2,dep/6,edge/6,rank/2,w2l/3,svo/4,summary/2,keyword/1]
    ),
    abolish(FN)
  )),
  consult(F),
  retractall(old_answer(_)),
  nb_setval(fname,FNameNoSuf).

% test if a text file is loaded  
is_loaded(FNameNoSuf):-catch(nb_getval(fname,FNameNoSuf),_,fail).  
  
% answers a call from Python - trying a moderately clever
% search through the database, for the highest ranked sentences
% that match the query
% it expects query_dep, query_rank, query_sent for the parsed query
% all expected to be in agenerated file FNameNoSuf_query.pro
ask(Answer):-
  is_loaded(FNameNoSuf),
  ask(FNameNoSuf,Answer).

% main interface predicate, !!!!!!!!!!!!!!
% Python/pyswip queries are of this form
ask(FNameNoSuf,Answer):-
  atomic_list_concat([FNameNoSuf,'_query.pro'],F),
  do((
    member(FN,
     [query_sim/4,query_sent/2,query_ner/2,query_dep/6,query_edge/6,query_rank/2,
      query_w2l/3,query_svo/4,query_rel/4,query_summary/2,query_keyword/1,
      query_param/2,query_pers_sents/2,query_pers_words/2]
    ),
    abolish(FN)
  )),
  consult(F),
  query_param(max_answers,MaxAnswers),
  show_query,
  answer(MaxAnswers,Answer),
  ppp('!!! answer':Answer),
  true.

show_query:-
  listing(query_sent).
  /*
  do((
    query_sent(N,_),
    N>0,
    nice_sent(N,Sent),
    writeln(Sent)
  )),
 nl.
*/


% expands query after detecting what
% content resources are availble in the database
% e.g., relations, similarities etc.
expanded_query_rank(W,WR):-
    query_rank(XL,R),
    %once((query_w2l(_,XL,XT),good_tag(XT))),
    in_focus(XL,XT),
    ( W=XL,WR=R
    ; is_defined(plugin_rel/4),
      ( plugin_rel(XL,_Rel,W,Amplifier)
      ; plugin_rel(W,_Rel,XL,Amplifier)
      ),
      rank(W,WR0),
      WR is Amplifier*WR0
    ; is_defined(query_rel/4),
      query_rel(XL,_,W,_),
      rank(W,WR)
    ; 
      is_defined(query_sim/4),
      query_sim(XL,XT,W,WT),
      good_tag(WT),    
      rank(W,RR),WR is R*RR
    ).

in_focus(W,T):-distinct(in_focus_or_context(W,T)).

in_focus_or_context(W,WT):-
  by_count(W0-WT0,in_focus_at(0,W0,WT0)),
  ( W-WT = W0-WT0
  ; by_count(W-WT,in_query_context(W0,WT0,W,WT))
  ).

in_query_context(W0,WT0,W,WT):-
   in_focus_at(W0,WT0,SentId),
   SentId>0,
   in_focus_at(W,WT,SentId),
   W0\==W.


in_focus_at(SentId,W,WT):-
  query_edge(SentId, F, FT, _, T, TT),
  ( good_tag(FT),W=F,WT=FT
  ; good_tag(TT),W=T,WT=TT
  ).

% collects sentence id matches derived from SVO relations  
match_svo(K):-
  is_defined(svo/4),
  is_defined(query_svo/4),
  query_svo(S,V,O,_),
  ( svo(S,V,O,K)
  ; svo(S,_,O,K)
  ; svo(S,V,_,K)
  ; svo(_,V,O,K)
%  ; svo(S,_,_,K)
%  ; svo(_,_,O,K)
  ).
match_svo(I):-
  is_defined(svo/4),
  distinct(in_focus_at(0,W,WT)),
  good_tag(WT),
  ( W=S,svo(S,_,_,I)
  ; W=O,svo(_,_,O,I)
  ).
  
% collects sentence id by matching agains edges
match_edges(K):-
  is_defined(query_edge/6),
  is_defined(edge/6),
  distinct(K,lift_query_edge(K)).

% replaces wh-words with vriables
% then tries matching edges
% generates sentence ids K that match
lift_query_edge(K):-
  query_edge(0, F,FT, Label, T, TT),
  good_tag(TT),FT\=='SENT',
  ( wh_tag(FT)->LiftedE=edge(K, _F,FT1, _L, T, TT)
  ; LiftedE=edge(K, F,FT1, Label, T, TT),FT=FT1
  ),
  call(LiftedE),
  good_tag(FT1,'N').
  %writeln(F1+FT1=LiftedE).

personalize:-
  query_param('personalize', Count),Count>0,
  is_defined(query_pers_sents/2).


% collects MaxAns highest ranked answers
answer(MaxAns,Answer):-
  findnsols(MaxAns,Id-Algo,fresh_answer(Id,Algo),Ids),
  !,
  build_answer(Ids,Answer).

fresh_answer(Id,Algo):-
  search_answer(Id,Algo),
  ( query_param(repeat_answers,yes)->true
  ; old_answer(Id)->fail
  ; assertz(old_answer(Id))
  ).

% builds a stream of answers using several search algorithms


search_answer(Id,Algo):-
   personalize,
   !,
   distinct(Id,search_answer0(Id,Algo)),
   query_pers_sents(Id,_).
search_answer(SentId,Algo):-
  distinct(SentId,search_answer0(SentId,Algo)).

search_answer0(SentId,ner):-distinct(match_ners(SentId)).
search_answer0(SentId,relevant):-distinct(match_relevant(SentId)).
search_answer0(SentId,edges):-distinct(match_edges(SentId)).
search_answer0(SentId,svo):-distinct(match_svo(SentId)).

% matches using ranks in both query and matching data
match_relevant(SentId):-
   % best by rank in query
   findnsols(60,R-X,(
       distinct(expanded_query_rank(X,QR)), % lemma
       R=QR    
     ),
     RankedXs
   ),!, 
   % best by weights coming from joint ranks
   revsort(RankedXs,RXs),
 
   % drop ranks,
   maplist(arg(2),RXs,Xs), % highest ranked matching words
   % find matching sentence ids and their ranks  
   findall(R-N,
     ( sent(N,Ws),
       maplist(to_lemma,Ws,Ls),
       append(Ws,Ls,WLs0),sort(WLs0,WLs),
       intersection(Xs,WLs,Is),
       length(Is,L),L>=2,
       findall(R,(member(I,Is),member(R-I,RXs)),Rs),    
       sumlist(Rs,R0),R is L*R0
     ),
     RNs
   ),
   % sort by decreasing ranks
   revsort(RNs,Ranked),
   member(_-SentId,Ranked).
   
   % pick the best MaxAns sentence ids
   %findnsols(MaxAns,N,member(_-N,Ranked),Ids),
   %!.
   


build_answer(SentIds,Answer):-
     % sort them in the order they occur in document - maybe?
   ( query_param(by_rank,'yes') -> SortedNs=SentIds
   ; sort(SentIds,SortedNs)
   ),
   % fuse words associated to sentence ids into atomic answers
   member(N-Algo,SortedNs),
   nice_sent(N,Sent),
   ppp(algorithm_for(N,Algo)),
   Answer=Sent.

nice_sent(N,Sent):-
  sent(N,Ws1),
  subst('-LSB-','(',Ws1,Ws2),
  subst('-RSB-',')',Ws2,Ws3),
   subst('-LRB-','(',Ws3,Ws4),
  subst('-RRB-',')',Ws4,Ws),
  intersperse([N,':'|Ws],' ',SWs),
  atomic_list_concat(SWs,Sent).

good_tag(WT):-good_tag(WT,_).

good_tag(WT,C):-atom_chars(WT,[C|_]),memberchk(C,['N','V','J']).

wh_tag(WT):-atom_chars(WT,[C|_]),memberchk(C,['W']).

to_lemma(W,L):-once(w2l(W,L,_)).

subst(_,_,[],[]):-!.
subst(X,Y,[X|Xs],[Y|Ys]):-!,subst(X,Y,Xs,Ys).
subst(X,Y,[A|Xs],[A|Ys]):-subst(X,Y,Xs,Ys).   

% intersperses words Ws with separator X
intersperse(Ws,X,Ys):-intersperse(Ws,X,Ys,[]).

intersperse([],_)-->[].
intersperse([W|Ws],X)-->[W,X],intersperse(Ws,X).

intersperse0(Ws,X,Ys0):-intersperse(Ws,X,Ys),once(append(Ys0,[_],Ys)).

% generates variabts of a word
% Capitalizes, stems and drops on char if lonerg than 4

word_variant_of(W,V):-
  lex_variant_of(W,V)
;
  word_misspelling_of(W,V)
;
  downcase_atom(W,LowerW),
  word_misspelling_of(LowerW,V).


lex_variant_of(W,V):-
  downcase_atom(W,LowerW),
  atom_codes(W,Cs),
  %maplist(to_lower,Cs,Ls),atom_codes(LowerW,Ls),
  Cs=[L|Ds],to_upper(L,U),
  Us=[U|Ds],atom_codes(Capitalized,Us),
  %porter_stem(W,Stem),
  snowball(LowerW,Stem),
  sort([W,LowerW,Capitalized,Stem],Ws),
  member(V,Ws).

word_misspelling_of(W,MissinCharInW):-
  atom_codes(W,Cs),
  length(Cs,Len),Len>4, % only for long enough words ...
  ( select(_,Cs,Ms)
  ; transp_letters_in(Cs,Ms)
  ),
  atom_codes(MissinCharInW,Ms).


transp_letters_in([X,Y|Xs],[Y,X|Xs]).
transp_letters_in([Y|Xs],[Y|Ys]):-transp_letters_in(Xs,Ys).

snowball(W,Stem):-
    snowball(english,W,Stem).

% sorts in decreasing order
revsort(Xs,Decreasing):-sort(0,(@>),Xs,Decreasing).

revkeysort(Xs,Decreasing):-sort(1,(@>=),Xs,Decreasing).

keygroup(Ps,Gs):-
 keysort(Ps,Qs),
 group_pairs_by_key(Qs,Gs).
 

by_count(X,G):-
  findall(X,call(G),Ms),
  by_count_all(Ms,Xs),
  member(X,Xs).

by_count_all(Ms,Xs):-
  count_occs(Ms,KXs),
  maplist(arg(2),KXs,Xs).

count_occs(Ms,KXs):-
  maplist(add_mark,Ms,Xs),
  keygroup(Xs,XMs),
  maplist(count_marks,XMs,CXs),
  revkeysort(CXs,KXs).

add_mark(M,M-1).
count_marks(M-Xs,K-M):-sum_list(Xs,K).

freqsort(MultiSet,Set):-
  findall(M-1,member(M,MultiSet),MUs),keysort(MUs,MVs),
  group_pairs_by_key(MVs,Grouped),
  maplist(by_len,Grouped,LXs),
  keysort(LXs,RSet),
  reverse(RSet,Set).

most_freq_of(MultiSet,Elem):-freqsort(MultiSet,[_-Elem|_]).

by_len(K-Us,L-K):-length(Us,L).


% runs all goals in Gs, but just for their side effects
do(Gs):-Gs,fail;true.

is_defined(F/N):-
  functor(T,F,N),
  predicate_property(T,defined),
  !.

% counts number of clauses of given predicate F/N
cls_count(F/N,K):-functor(C,F,N),predicate_property(C, number_of_clauses(K)).

% NER logic

match_ners(S):-is_defined(match_custom_ners/1),match_custom_ners(S).
match_ners(S):-
  is_defined(ner/2),
  once((query_w2l(_,L,_),wh_word(L))),
  findall(W,(query_w2l(W,_L,Tag),good_tag(Tag)),Ws),
  %writeln(Ws),
  call(L,Ws,S).



wh_word(where).
wh_word(when).
wh_word(who).
wh_word(many).



who(S):-who([_],S).
who(KWs,S):-wh(['PERSON','ORGANIZATION','TITLE'],KWs,S).

many(S):-many([_],S).
many(KWs,S):-wh(['NUMBER', 'ORDINAL', 'MONEY'],KWs,S).

when(S):-when([_],S).
when(KWs,S):-wh(['DATE','TIME','DURATION'],KWs,S).

where(S):-where([_],S).
where(KWs,S):-wh(['LOCATION','CITY','COUNTRY','STATE_OR_PROVINCE'],KWs,S).

wh(Tags,S):-wh(Tags,[_],S).
wh(Tags,KWs,S):-distinct(S,wh0(Tags,KWs,S)).

wh0(Tags,KWs,S):-
  ner(S,Ps),
  member(Tag,Tags),
  member((_Pos,_LW,Tag),Ps),
  sent(S,Ws),
  shares_kwords(1,KWs,Ws,_).

shares_kwords(K,Us,Vs,Is):-
  intersection(Us,Vs,Is),
  length(Is,L),L>=K.


% transitive closure for relational reasoning


call_svo(A,Rel,B,Id):-svo(A,Rel,B,Id);svo(B,Rel,A,Id).

tc_search(K,Word,Id):-
  length(Rels,K),
  tc_search(K,Word,Rels,Id).

tc_search(K,Word,Rels,Id):-
  must_be(list,Rels),
  distinct(Id,(
      tc(K,Word,Rels,_RelatedWord,res(_Steps,Id,_Path))
    )
  ).

tc(K,A,Rels,C,Res):-tc(A,Rels,C,[],K,_,Res).

tc(A,Rels,C,Xs,SN1,N2,Res) :-
  succ(N1,SN1),
  member(Rel,Rels),
  call_svo(A,Rel,B,Id),
  not(memberchk(B-_,Xs)),
  tc1(B,Rels,C,[A-Rel|Xs],Id,N1,N2,Res).

tc1(B,_Rels,B,Xs,Id,N,N,res(N,Id,Xs)):-nonvar(Id).
tc1(B,Rels,C,Xs,_,N1,N2,Res) :-
   tc(B,Rels,C,Xs,N1,N2,Res).

tc2res(K,A,Rels,B,res(Steps,Id,Path)):-
  distinct(A+B+End+Id,tc(A,Rels,B,[],K,End,res(N,Id,RevPath))),
  Steps is K-N,
  reverse(RevPath,Path).



% testing for all answers

test(FNameNoSuf):-
  load(FNameNoSuf),
  do((
    ask(FNameNoSuf,A),
    writeln(A)
  )).

test:-is_loaded(FNameNoSuf),test(FNameNoSuf).

ppp(X):-trace(T),T>0->writeln(X);true.

t0:-test('examples/tesla').
t1:-test('examples/bfr').
t2:-test('examples/hindenburg').
t3:-test('examples/const').
t4:-test('examples/summary').
t5:-test('examples/heaven').
t6:-test('examples/einstein').
t7:-test('examples/kafka').
t8:-test('examples/test').
t9:-test('examples/relativity').
t10:-test('examples/textrank').
t11:-test('examples/texas').
t12:-test('examples/heli').
t13:-test('examples/red').


all_ts:- do((
  between(0,13,I),
  atom_concat(t,I,G),
  writeln('-----------------------------------'),
  writeln(G),
  call(G)
  )).
