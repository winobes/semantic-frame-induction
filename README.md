# TODO

* Paper
    * Introduction/background
    * Related work
    * Models we tried / derivations?
    * Results
    * Discussion
    * Conclusion
* Presentation
    * ~~what are frames example~~ 
    * model slides
    * evaluation
* Implementation
    * ~~Model 0 (EM)~~
    * Adapted Model 1 (Gibbs)
        * ~~derive Gibbs~~
        * ~~implement~~
        * optimize if possible
        * fix the in/out for consistency with EM
* Results
    * DiceSim with framenet
    * ~~Frame coherency~~
    * compare performance per verb categories
    
# Plan!

## Stage 1

- Derive EM for model 0 (as in the Rooth paper)
- Get the data ready for this (verbargs from google n-gram)
- implement the EM
- testing our implementation
    - use some measures (e.g. DiceSim to framenet)
        * frame coherency
        * DiceSim to framenet
    - maybe try to find Gold Standard data
    - compare to other semantic frame induction (Ivan's, CMU?)
    - find areas of improvement

## Stage 2 

- adapt the model
- repeat steps in stage 1
