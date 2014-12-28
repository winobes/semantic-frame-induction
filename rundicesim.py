
import dicesim
import pickle

import framenet

##------------------------

#sample_filename = 'sample_m1_F50_alpha0.5_beta0.5_T500burnIn50.pkl'
sample_filename = 'sample_m1_F200_alpha0.5_beta0.5_T1000burnIn50.pkl'
#mvmc = 'topN'
#mvmc_param = 4    # for topN
mvmc = 'cutoffprob'
mvmc_param = .01  # for cutoffprob
fn_threshold = 5


dice_scoresheet, dice_assign = dicesim.run_dicesim(sample_filename,
                                                   mvmc, mvmc_param,
                                                   fn_threshold)


#dice_results = dicesim.compile_report( ... )
#dicesim.print_dice_results(dice_results)

                  

    
##------------------------

    
    # todo:
    #   - experiment with the "model_verb_membership_criterion" and its params
    #   - print output/scores informatively
    #   - integrate framenet info into output
    #   - restructure these code files better
    #   - rename some variables and functions for clarity
    #   - some other stuff

    # potentially:
    #   - modify the framenet interface code so labels are ints, not 'int's
    #   - use sets instead of lists in some places

    # questions:
    #   - what does the distribution of dicesim scores for a fixed FN frame
    #     over all of our model's frames look like? Is there a lot of verb-overlap
    #     between the different frames induced by our own model?

