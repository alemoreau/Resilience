(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "twoside")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "a4paper") ("inputenc" "latin1") ("fontenc" "T1") ("algorithm2e" "linesnumbered" "ruled") ("cite" "biblabel" "nomove") ("caption" "labelfont={bf,sf}" "labelsep=period" "justification=raggedright") ("hyperref" "colorlinks=true" "allcolors=blue")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "geometry"
    "amsmath"
    "amsfonts"
    "amssymb"
    "inputenc"
    "fontenc"
    "pgf"
    "tikz"
    "RR"
    "flexisym"
    "algorithm2e"
    "subcaption"
    "cite"
    "caption"
    "hyperref"
    "xcolor"
    "float"
    "amsthm")
   (TeX-add-symbols
    "allOne")
   (LaTeX-add-labels
    "eq:minNomr"
    "alg:gmres"
    "alg:precond-gmres"
    "alg:faulty_product"
    "fig:gre_216a"
    "fig:pores_2"
    "table:nonlin"
    "sec:empirical"
    "fig:gre_216a_conv_hist"
    "table:outcomes"
    "fig:gre_216a_conv_hist_bit_0"
    "fig:gre_216a_conv_hist_bit_1"
    "fig:gre_216a_conv_hist_bit_2"
    "fig:pores_2_conv_hist_bit_0"
    "fig:pores_2_conv_hist_bit_1"
    "fig:pores_2_conv_hist_bit_2"
    "fig:conv_hist_bit"
    "fig:gre_216a_conv_hist_iteration_0"
    "fig:gre_216a_conv_hist_iteration_1"
    "fig:gre_216a_conv_hist_iteration_2"
    "fig:pores_2_conv_hist_iteration_0"
    "fig:pores_2_conv_hist_iteration_1"
    "fig:pores_2_conv_hist_iteration_2"
    "fig:conv_hist_iteration"
    "fig:gre_216a_conv_hist_register_0"
    "fig:gre_216a_conv_hist_register_1"
    "fig:gre_216a_conv_hist_register_2"
    "fig:pores_2_conv_hist_register_0"
    "fig:pores_2_conv_hist_register_1"
    "fig:pores_2_conv_hist_register_2"
    "fig:conv_hist_register"
    "fig:gre_216a_conv_hist_location_0"
    "fig:gre_216a_conv_hist_location_1"
    "fig:gre_216a_conv_hist_location_2"
    "fig:pores_2_conv_hist_location_0"
    "fig:pores_2_conv_hist_location_1"
    "fig:pores_2_conv_hist_location_2"
    "fig:conv_hist_location"
    "fig:gre_216a_bit_iteration_0"
    "fig:pores_2_bit_iteration_0"
    "fig:gre_216a_bit_iteration_1"
    "fig:pores_2_bit_iteration_1"
    "fig:bit_iteration"
    "fig:bit_iteration_0"
    "fig:bit_iteration_1"
    "fig:matrices_bit_iteration"
    "sec:propagation"
    "fig:error_propagation"
    "sec:quantify"
    "fig:gre_216a_conv_hist_delta"
    "fig:pores_2_conv_hist_delta"
    "inexact_krylov"
    "inexact_krylov_adapted"
    "eqn:delta"
    "sec:prediction"
    "theorem"
    "scheme_oracle"
    "fig:gre_216a_conv_hist_prediction"
    "fig:gre_216a_conv_hist_no_prediction"
    "fig:pores_2_conv_hist_prediction"
    "fig:pores_2_conv_hist_no_prediction"
    "fig:compared"
    "fig:prediction"
    "sec:faultDetection"
    "sec:detection_oracle"
    "detection_scheme_oracle"
    "table:theoretical_outcomes"
    "sec:evaluation_oracle"
    "fig:gre_216a_test_result_oracle_0"
    "fig:gre_216a_test_result_oracle_1"
    "fig:pores_2_test_result_oracle_0"
    "fig:pores_2_test_result_oracle_1"
    "fig:test_result_oracle"
    "fig:gre_216a_test_result_c05_oracle_0"
    "fig:gre_216a_test_result_c05_oracle_1"
    "fig:pores_2_test_result_c05_oracle_0"
    "fig:pores_2_test_result_c05_oracle_1"
    "fig:test_result_oracle_c05"
    "sec:checksum"
    "eqn:checksum"
    "fig:gre_216a_conv_hist_checksum_0"
    "fig:gre_216a_conv_hist_checksum_1"
    "fig:gre_216a_conv_hist_checksum_2"
    "fig:pores_2_conv_hist_checksum_0"
    "fig:pores_2_conv_hist_checksum_1"
    "fig:pores_2_conv_hist_checksum_2"
    "fig:conv_hist_checksum"
    "sec:threshold"
    "fig:gre_216a_conv_hist_threshold_evaluation_0"
    "fig:gre_216a_conv_hist_threshold_evaluation_1"
    "fig:pores_2_conv_hist_threshold_evaluation_0"
    "fig:pores_2_conv_hist_threshold_evaluation_1"
    "fig:conv_hist_threshold_evaluation"
    "sec:practical_scheme"
    "colors"
    "fig:gre_216a_conv_hist_threshold_0"
    "fig:gre_216a_conv_hist_threshold_2"
    "fig:pores_2_conv_hist_threshold_0"
    "fig:pores_2_conv_hist_threshold_2"
    "fig:conv_hist_threshold"
    "fig:gre_216a_conv_hist_threshold_3"
    "fig:gre_216a_conv_hist_threshold_4"
    "fig:pores_2_conv_hist_threshold_3"
    "fig:pores_2_conv_hist_threshold_4"
    "fig:conv_hist_threshold_false"
    "sec:implementable_evaluated"
    "fig:gre_216a_test_result_0"
    "fig:gre_216a_test_result_1"
    "fig:pores_2_test_result_0"
    "fig:pores_2_test_result_1"
    "fig:test_result"
    "fig:gre_216a_test_result_c05_0"
    "fig:gre_216a_test_result_c05_1"
    "fig:pores_2_test_result_c05_0"
    "fig:pores_2_test_result_c05_1"
    "fig:test_result_c05"
    "fig:why"
    "fig:test_result_c05_0_full"
    "fig:test_result_c05_1_full"
    "fig:test_result_c05_0_precond"
    "fig:test_result_c05_1_precond"
    "fig:test_result_c05_matrices")
   (LaTeX-add-bibliographies
    "sample")
   (LaTeX-add-counters
    "fig"
    "grecounter"
    "porescounter")
   (LaTeX-add-xcolor-definecolors
    "50, 150, 50"
    "blue"
    "red"
    "30, 30, 30"
    "85, 147, 47"
    "90, 90, 90"
    "orange")
   (LaTeX-add-amsthm-newtheorems
    "theorem"))
 :latex)

