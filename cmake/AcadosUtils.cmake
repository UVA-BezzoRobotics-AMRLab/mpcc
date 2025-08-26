# Helper to collect acados generated code
function(collect_acados_generated_code model_prefix)
  set(output_dir ${CMAKE_SOURCE_DIR}/scripts/${model}/c_generated_code)

  set(${model_prefix}_SRC
    ${output_dir}/acados_solver_unicycle_model_mpcc.c
    ${output_dir}/acados_sim_solver_unicycle_model_mpcc.c
    ${output_dir}/${model}_model/${model}_expl_ode_fun.c
    ${output_dir}/${model}_model/${model}_expl_vde_adj.c
    ${output_dir}/${model}_model/${model}_expl_vde_forw.c
    ${output_dir}/${model}_model/${model}_expl_ode_hess.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_fun.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_0_fun.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_e_fun.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_fun_jac.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_0_fun_jac.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_e_fun_jac.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_fun_jac_hess.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_0_fun_jac_hess.c
    ${output_dir}/${model}_cost/${model}_cost_ext_cost_e_fun_jac_hess.c
  )

  set(${model_prefix}_HEADERS
    ${output_dir}/acados_solver_${model}.h
    ${output_dir}/acados_sim_solver_${model}.h
    ${output_dir}/${model}_cost/${model}_cost.h
    ${output_dir}/${model}_model/${model}_model.h
  )

  set(${model_prefix}_LIBS
    ${output_dir}/libacados_ocp_solver_${model}.so
    ${output_dir}/libacados_sim_solver_${model}.so
  )

  set(${model_prefix}_OUTPUT_FILES 
    ${${model_prefix}_SRC} 
    ${${model_prefix}_HEADERS}
    ${${model_prefix}_LIBS}
    PARENT_SCOPE
  )

  set(${model_prefix}_SRC "${${model_prefix}_SRC}" PARENT_SCOPE)
  set(${model_prefix}_HEADERS "${${model_prefix}_HEADERS}" PARENT_SCOPE)
  set(${model_prefix}_LIBS "${${model_prefix}_LIBS}" PARENT_SCOPE)

endfunction()

