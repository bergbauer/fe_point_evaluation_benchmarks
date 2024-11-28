
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/numerics/vector_tools.h>

#define FE_EVAL
#define FE_POINT
#define FE_POINT_VEC
// #define FE_POINT_MANUAL
// #define FE_VAL
// #define SYSTEM_MATRIX

#define SURFACE_TERMS 1

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

static constexpr unsigned int n_lanes = dealii::VectorizedArray<double>::size();

template <int dim>
void
test_cg(const unsigned int degree, const unsigned int n_dofs)
{
  using namespace dealii;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  // calculate subdivisions/refinements
  const unsigned int n_cells = n_dofs / Utilities::fixed_power<dim>(degree);

  const unsigned int child_cells_per_cell =
    ReferenceCells::get_hypercube<dim>().n_isotropic_children();

  unsigned int n_global_refinements = 0;
  unsigned int n_subdivisions       = 0;
  double       cells_on_coarse_grid = n_cells;
  while (cells_on_coarse_grid > 8000)
    {
      cells_on_coarse_grid /= child_cells_per_cell;
      n_global_refinements++;
    }

  if (dim == 2)
    n_subdivisions = std::ceil(std::sqrt(cells_on_coarse_grid));
  else if (dim == 3)
    n_subdivisions = std::ceil(std::cbrt(cells_on_coarse_grid));
  else
    AssertThrow(false, ExcNotImplemented());

  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);
  tria.refine_global(n_global_refinements);

  FE_Q<dim>       fe(degree);
  FE_DGQ<dim>     fe_dg(fe.degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  MappingQGeneric<dim> mapping(1);

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  MatrixFree<dim> matrix_free;
  matrix_free.reinit(mapping,
                     dof_handler,
                     constraints,
                     QGauss<1>(degree + 1),
                     typename MatrixFree<dim>::AdditionalData());

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Working with " << fe.get_name() << " and "
              << dof_handler.n_dofs() << " dofs" << std::endl;

  LinearAlgebra::distributed::Vector<double> src, dst, dst2, dst3, dst4;
  matrix_free.initialize_dof_vector(src);
  for (auto &v : src)
    v = static_cast<double>(rand()) / RAND_MAX;

  matrix_free.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(dst2);
  matrix_free.initialize_dof_vector(dst3);
  matrix_free.initialize_dof_vector(dst4);

  unsigned int n_tests  = 100;
  double       min_time = std::numeric_limits<double>::max();
  double       max_time = 0;
  double       avg_time = 0;
#ifdef FE_EVAL
  for (unsigned int t = 0; t < n_tests; ++t)
    {
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("fe_evaluation_" + std::to_string(degree)).c_str());
#  endif

      Timer time;
      matrix_free
        .template cell_loop<LinearAlgebra::distributed::Vector<double>,
                            LinearAlgebra::distributed::Vector<double>>(
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEEvaluation<dim, -1> fe_eval(matrix_free);
            for (unsigned int cell = range.first; cell < range.second; ++cell)
              {
                fe_eval.reinit(cell);
                fe_eval.gather_evaluate(src, EvaluationFlags::gradients);
                for (const unsigned int q : fe_eval.quadrature_point_indices())
                  fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
                fe_eval.integrate_scatter(EvaluationFlags::gradients, dst);
              }
          },
          dst,
          src,
          true);
      const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("fe_evaluation_" + std::to_string(degree)).c_str());
#  endif
      min_time = std::min(min_time, tw);
      max_time = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "FEEvaluation test      in " << std::setw(11) << min_time
              << " " << std::setw(11) << avg_time << " " << std::setw(11)
              << max_time << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

  QGauss<dim>                  quad(degree + 1);
  std::vector<Quadrature<dim>> quad_vec;
  quad_vec.reserve(matrix_free.n_cell_batches() * n_lanes);

#if SURFACE_TERMS
  QGauss<dim - 1>         quad_dim_m1(degree + 1);
  std::vector<Point<dim>> points(quad_dim_m1.size());
  for (unsigned int i = 0; i < quad_dim_m1.size(); ++i)
    {
      for (unsigned int d = 0; d < dim - 1; ++d)
        points[i][d] = quad_dim_m1.get_points()[i][d];
      points[i][dim - 1] = 0.5;
    }

  std::vector<Tensor<1, dim>> normals(quad_dim_m1.size());
  for (auto &n : normals)
    n[dim - 1] = 1.;
  NonMatching::ImmersedSurfaceQuadrature<dim> quad_surface(
    points, quad_dim_m1.get_weights(), normals);
  std::vector<NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
  quad_vec_surface.reserve(matrix_free.n_cell_batches() * n_lanes);
#endif

  std::vector<typename DoFHandler<dim>::cell_iterator> vector_accessors;
  vector_accessors.reserve(matrix_free.n_cell_batches() * n_lanes);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    for (unsigned int v = 0; v < n_lanes; ++v)
      {
        if (v < matrix_free.n_active_entries_per_cell_batch(cell))
          vector_accessors.push_back(matrix_free.get_cell_iterator(cell, v));
        else
          vector_accessors.push_back(matrix_free.get_cell_iterator(cell, 0));

        quad_vec.push_back(quad);
#if SURFACE_TERMS
        quad_vec_surface.push_back(quad_surface);
#endif
      }

  min_time = std::numeric_limits<double>::max();
  max_time = 0;
  avg_time = 0;
  n_tests  = 100;
#ifdef FE_POINT
  {
    NonMatching::MappingInfo<dim> mapping_info(mapping,
                                               update_gradients |
                                                 update_JxW_values);

    mapping_info.reinit_cells(vector_accessors, quad_vec);

    FEPointEvaluation<1, dim, dim, double> fe_peval(mapping_info, fe_dg);

#  if SURFACE_TERMS
    NonMatching::MappingInfo<dim> mapping_info_surface(mapping,
                                                       update_values |
                                                         update_gradients |
                                                         update_JxW_values |
                                                         update_normal_vectors);

    mapping_info_surface.reinit_surface(vector_accessors, quad_vec_surface);

    FEPointEvaluation<1, dim, dim, double> fe_peval_surface(
      mapping_info_surface, fe_dg);
#  endif
    for (unsigned int t = 0; t < n_tests; ++t)
      {
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_START(("fe_point_" + std::to_string(degree)).c_str());
#  endif
        Timer time;
        matrix_free
          .template cell_loop<LinearAlgebra::distributed::Vector<double>,
                              LinearAlgebra::distributed::Vector<double>>(
            [&](const auto &matrix_free,
                auto &      dst,
                const auto &src,
                const auto &range) {
              FEEvaluation<dim, -1> fe_eval(matrix_free);
              for (unsigned int cell = range.first; cell < range.second; ++cell)
                {
                  fe_eval.reinit(cell);
                  fe_eval.read_dof_values(src);
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    {
                      fe_peval.reinit(cell * n_lanes + v);
                      fe_peval.evaluate(StridedArrayView<const double, n_lanes>(
                                          &fe_eval.begin_dof_values()[0][v],
                                          fe_eval.dofs_per_cell),
                                        EvaluationFlags::gradients);
                      for (const unsigned int q :
                           fe_peval.quadrature_point_indices())
                        fe_peval.submit_gradient(fe_peval.get_gradient(q), q);
                      fe_peval.integrate(StridedArrayView<double, n_lanes>(
                                           &fe_eval.begin_dof_values()[0][v],
                                           fe_eval.dofs_per_cell),
                                         EvaluationFlags::gradients);

#  if SURFACE_TERMS
                      fe_peval_surface.reinit(cell * n_lanes + v);
                      fe_peval_surface.evaluate(
                        StridedArrayView<const double, n_lanes>(
                          &fe_eval.begin_dof_values()[0][v],
                          fe_eval.dofs_per_cell),
                        EvaluationFlags::gradients | EvaluationFlags::values);
                      for (const unsigned int q :
                           fe_peval_surface.quadrature_point_indices())
                        {
                          const auto &value = fe_peval_surface.get_value(q);
                          const auto &gradient =
                            fe_peval_surface.get_gradient(q);

                          fe_peval_surface.submit_value(
                            fe_peval_surface.normal_vector(q) * gradient * 0.,
                            q);
                          fe_peval_surface.submit_gradient(
                            fe_peval_surface.normal_vector(q) * value * 0., q);
                        }
                      fe_peval_surface.integrate(
                        StridedArrayView<double, n_lanes>(
                          &fe_eval.begin_dof_values()[0][v],
                          fe_eval.dofs_per_cell),
                        EvaluationFlags::gradients | EvaluationFlags::values,
                        true);
#  endif
                    }
                  fe_eval.distribute_local_to_global(dst);
                }
            },
            dst2,
            src,
            true);
        const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP(("fe_point_" + std::to_string(degree)).c_str());
#  endif
        min_time = std::min(min_time, tw);
        max_time = std::max(max_time, tw);
        avg_time += tw / n_tests;
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation test in " << std::setw(11) << min_time
                << " " << std::setw(11) << avg_time << " " << std::setw(11)
                << max_time << " seconds, throughput " << std::setw(8)
                << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
                << std::endl;
  }
#endif

  min_time = std::numeric_limits<double>::max();
  max_time = 0;
  avg_time = 0;
  n_tests  = 100;

#ifdef FE_POINT_VEC
  NonMatching::MappingInfo<dim, dim, VectorizedArray<double>> mapping_info_vec(
    mapping, update_gradients | update_JxW_values);

  mapping_info_vec.reinit_cells(vector_accessors, quad_vec);

  FEPointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval_vec(
    mapping_info_vec, fe_dg);

#  if SURFACE_TERMS
  NonMatching::MappingInfo<dim, dim, VectorizedArray<double>>
    mapping_info_surface_vec(mapping,
                             update_values | update_gradients |
                               update_JxW_values | update_normal_vectors);

  mapping_info_surface_vec.reinit_surface(vector_accessors, quad_vec_surface);

  FEPointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval_surface_vec(
    mapping_info_surface_vec, fe_dg);
#  endif

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      Vector<double> solution_values(fe.dofs_per_cell);
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("fe_point_vec_" + std::to_string(degree)).c_str());
#  endif
      Timer time;
      matrix_free.template cell_loop<
        LinearAlgebra::distributed::Vector<double>,
        LinearAlgebra::distributed::Vector<double>>(
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            const auto &range) {
          FEEvaluation<dim, -1> fe_eval(matrix_free);
          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              fe_eval.reinit(cell);
              fe_eval.read_dof_values(src);
              for (unsigned int v = 0; v < n_lanes; ++v)
                {
                  fe_peval_vec.reinit(cell * n_lanes + v);
                  fe_peval_vec.evaluate(StridedArrayView<const double, n_lanes>(
                                          &fe_eval.begin_dof_values()[0][v],
                                          fe_eval.dofs_per_cell),
                                        EvaluationFlags::gradients);
                  for (const unsigned int q :
                       fe_peval_vec.quadrature_point_indices())
                    fe_peval_vec.submit_gradient(fe_peval_vec.get_gradient(q),
                                                 q);
                  fe_peval_vec.integrate(StridedArrayView<double, n_lanes>(
                                           &fe_eval.begin_dof_values()[0][v],
                                           fe_eval.dofs_per_cell),
                                         EvaluationFlags::gradients);

#  if SURFACE_TERMS
                  fe_peval_surface_vec.reinit(cell * n_lanes + v);
                  fe_peval_surface_vec.evaluate(
                    StridedArrayView<const double, n_lanes>(
                      &fe_eval.begin_dof_values()[0][v], fe_eval.dofs_per_cell),
                    EvaluationFlags::gradients | EvaluationFlags::values);
                  for (const unsigned int q :
                       fe_peval_surface_vec.quadrature_point_indices())
                    {
                      const auto &value = fe_peval_surface_vec.get_value(q);
                      const auto &gradient =
                        fe_peval_surface_vec.get_gradient(q);

                      fe_peval_surface_vec.submit_value(
                        fe_peval_surface_vec.normal_vector(q) * gradient * 0.,
                        q);
                      fe_peval_surface_vec.submit_gradient(
                        fe_peval_surface_vec.normal_vector(q) * value * 0., q);
                    }
                  fe_peval_surface_vec.integrate(
                    StridedArrayView<double, n_lanes>(
                      &fe_eval.begin_dof_values()[0][v], fe_eval.dofs_per_cell),
                    EvaluationFlags::gradients | EvaluationFlags::values,
                    true);
#  endif
                }
              fe_eval.distribute_local_to_global(dst);
            }
        },
        dst2,
        src,
        true);
      const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("fe_point_vec_" + std::to_string(degree)).c_str());
#  endif
      min_time = std::min(min_time, tw);
      max_time = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "FEPointEvaluation (vectorized) test in " << std::setw(11)
              << min_time << " " << std::setw(11) << avg_time << " "
              << std::setw(11) << max_time << " seconds, throughput "
              << std::setw(8) << 1e-6 * dof_handler.n_dofs() / avg_time
              << " MDoFs/s" << std::endl;
#endif

#ifdef FE_POINT_MANUAL
  min_time = std::numeric_limits<double>::max();
  max_time = 0;
  avg_time = 0;
  n_tests  = 100;

  if (degree == 1)
    {
      min_time = std::numeric_limits<double>::max();
      max_time = 0;
      avg_time = 0;

      std::vector<std::array<unsigned int, (1 << dim)>> dof_indices_linear(
        matrix_free.n_cell_batches() * n_lanes);
      std::vector<unsigned int> numbers_mapping_info(
        matrix_free.n_cell_batches() * n_lanes, numbers::invalid_unsigned_int);
      std::vector<types::global_dof_index> dof_indices(
        dof_indices_linear[0].size());
      std::vector<unsigned char> regular_cell(matrix_free.n_cell_batches() *
                                              n_lanes);

      for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(c);
             ++v)
          {
            const auto dcell = matrix_free.get_cell_iterator(c, v);
            dcell->get_dof_indices(dof_indices);
            numbers_mapping_info[c * n_lanes + v] = dcell->active_cell_index();
            bool has_constraints                  = false;
            for (unsigned int i = 0; i < dof_indices.size(); ++i)
              if (constraints.is_constrained(dof_indices[i]))
                {
                  has_constraints = true;
                  dof_indices_linear[c * n_lanes + v][i] =
                    numbers::invalid_unsigned_int;
                }
              else
                dof_indices_linear[c * n_lanes + v][i] =
                  dst4.get_partitioner()->global_to_local(dof_indices[i]);
            if (has_constraints)
              regular_cell[c * n_lanes + v] = 2;
            else
              regular_cell[c * n_lanes + v] = 1;
          }

      const auto function =
        [&](const auto &, auto &dst, const auto &src, const auto &range) {
          constexpr int      n_lanes  = n_lanes;
          constexpr int      n_points = 1 << dim;
          const unsigned int lower    = range.first * n_lanes,
                             upper    = range.second * n_lanes;
          for (unsigned int cell = lower; cell < upper; ++cell)
            {
              std::array<double, n_points> solution_values = {};
              if (regular_cell[cell] == 0)
                continue;

              if (regular_cell[cell] == 1)
                for (unsigned int i = 0; i < n_points; ++i)
                  solution_values[i] =
                    src.local_element(dof_indices_linear[cell][i]);
              else
                for (unsigned int i = 0; i < n_points; ++i)
                  {
                    if (dof_indices_linear[cell][i] !=
                        numbers::invalid_unsigned_int)
                      solution_values[i] =
                        src.local_element(dof_indices_linear[cell][i]);
                    else
                      solution_values[i] = 0;
                  }

#  if 0
            const unsigned int index = cell;
            const unsigned int geometry_index =
              mapping_info_vec.template compute_geometry_index_offset<false>(
                index, 0);
            const bool affine_cell =
              mapping_info_vec.get_cell_type(geometry_index) <=
              internal::MatrixFreeFunctions::affine;
            const bool cartesian_cell =
              mapping_info_vec.get_cell_type(geometry_index) <=
              internal::MatrixFreeFunctions::cartesian;

            const unsigned int unit_point_offset =
              mapping_info_vec.compute_unit_point_index_offset(geometry_index);
            const auto unit_point_ptr =
              mapping_info_vec.get_unit_point(unit_point_offset);

            const unsigned int data_offset =
              mapping_info_vec.compute_data_index_offset(geometry_index);
            const unsigned int compressed_data_offset =
              mapping_info_vec.compute_compressed_data_index_offset(
                geometry_index);

            const auto inverse_jacobian_ptr =
              mapping_info_vec.get_inverse_jacobian(compressed_data_offset);
            const auto JxW_ptr = mapping_info_vec.get_JxW(data_offset);
            std::array<Tensor<1, dim, VectorizedArray<double>>,
                       n_points / n_lanes>
              grads;

            for (unsigned int q = 0; q < grads.size(); ++q)
              {
                const auto result =
                  internal::evaluate_tensor_product_value_and_gradient_linear(
                    solution_values.data(), unit_point_ptr[q]);
                Tensor<1, dim, VectorizedArray<double>> grad;
                for (unsigned int d = 0; d < dim; ++d)
                  grad[d] = result[d];
                grads[q] =
                  cartesian_cell ?
                    apply_diagonal_transformation(inverse_jacobian_ptr[0],
                                                  grad) :
                    apply_transformation(
                      inverse_jacobian_ptr[affine_cell ? 0 : q].transpose(),
                      grad);
              }

            std::array<VectorizedArray<double>, n_points>
              solution_values_vectorized;
            for (unsigned int q = 0; q < grads.size(); ++q)
              {
                grads[q] =
                  cartesian_cell ?
                    apply_diagonal_transformation(inverse_jacobian_ptr[0],
                                                  grads[q] * JxW_ptr[q]) :
                    apply_transformation(
                      inverse_jacobian_ptr[affine_cell ? 0 : q],
                      grads[q] * JxW_ptr[q]);
                VectorizedArray<double> value = 0;
                if (q == 0)
                  internal::
                    integrate_add_tensor_product_value_and_gradient_linear<
                      dim,
                      VectorizedArray<double>,
                      VectorizedArray<double>,
                      false>(&value,
                             grads[q],
                             solution_values_vectorized.data(),
                             unit_point_ptr[q]);
                else
                  internal::
                    integrate_add_tensor_product_value_and_gradient_linear<
                      dim,
                      VectorizedArray<double>,
                      VectorizedArray<double>,
                      true>(&value,
                            grads[q],
                            solution_values_vectorized.data(),
                            unit_point_ptr[q]);
              }
#  else
              fe_peval_vec.reinit(cell);
              fe_peval_vec.evaluate(solution_values,
                                    EvaluationFlags::gradients);
              for (const unsigned int q :
                   fe_peval_vec.quadrature_point_indices())
                fe_peval_vec.submit_gradient(fe_peval_vec.get_gradient(q), q);
              fe_peval_vec.integrate(solution_values,
                                     EvaluationFlags::gradients);
#  endif

              if (regular_cell[cell] == 1)
                for (unsigned int i = 0; i < n_points; ++i)
                  dst.local_element(dof_indices_linear[cell][i]) +=
                    solution_values[i];
              else
                for (unsigned int i = 0; i < n_points; ++i)
                  if (dof_indices_linear[cell][i] !=
                      numbers::invalid_unsigned_int)
                    dst.local_element(dof_indices_linear[cell][i]) +=
                      solution_values[i];
            }
        };
      for (unsigned int t = 0; t < n_tests; ++t)
        {
#  ifdef LIKWID_PERFMON
          LIKWID_MARKER_START(
            ("fe_point_manual_" + std::to_string(degree)).c_str());
#  endif
          Timer time;
          matrix_free
            .template cell_loop<LinearAlgebra::distributed::Vector<double>,
                                LinearAlgebra::distributed::Vector<double>>(
              function, dst4, src, true);
          const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
          LIKWID_MARKER_STOP(
            ("fe_point_manual_" + std::to_string(degree)).c_str());
#  endif
          min_time = std::min(min_time, tw);
          max_time = std::max(max_time, tw);
          avg_time += tw / n_tests;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "FEPointEvaluation (manual) test in " << std::setw(11)
                  << min_time << " " << std::setw(11) << avg_time << " "
                  << std::setw(11) << max_time << " seconds, throughput "
                  << std::setw(8) << 1e-6 * dof_handler.n_dofs() / avg_time
                  << " MDoFs/s" << std::endl;
    }
#endif

#ifdef FE_VAL
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      Timer                                time;
      QGauss<dim>                          quad(degree + 1);
      FEValues<dim>                        fe_values(mapping,
                              fe,
                              quad,
                              update_gradients | update_JxW_values);
      Vector<double>                       solution_values(fe.dofs_per_cell);
      std::vector<Tensor<1, dim>>          solution_gradients(quad.size());
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      src.update_ghost_values();
      dst3 = 0;

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(dof_indices);
            constraints.get_dof_values(src,
                                       dof_indices.begin(),
                                       solution_values.begin(),
                                       solution_values.end());
            fe_values.reinit(cell);
            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                Tensor<1, dim> gradient;
                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  gradient += solution_values(i) * fe_values.shape_grad(i, q);
                solution_gradients[q] = gradient * fe_values.JxW(q);
              }
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              {
                double sum = 0;
                for (unsigned int q = 0; q < quad.size(); ++q)
                  sum += solution_gradients[q] * fe_values.shape_grad(i, q);
                solution_values(i) = sum;
              }
            constraints.distribute_local_to_global(solution_values,
                                                   dof_indices,
                                                   dst3);
          }
      dst3.compress(VectorOperation::add);
      src.zero_out_ghost_values();
      const double tw = time.wall_time();
      min_time        = std::min(min_time, tw);
      max_time        = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "FEValues test          in " << std::setw(11) << min_time
              << " " << std::setw(11) << avg_time << " " << std::setw(11)
              << max_time << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

#ifdef SYSTEM_MATRIX
  const auto locally_owned_dofs = dof_handler.locally_owned_dofs();
  const auto locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);
  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);

  TrilinosWrappers::SparseMatrix system_matrix;

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       MPI_COMM_WORLD);

  FEValues<dim> fe_values(fe,
                          quad,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quad.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Timer time;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0.;

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix);
      }

  system_matrix.compress(VectorOperation::add);
  const double tw = time.wall_time();

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Assembly time: " << tw << std::endl;

  min_time = std::numeric_limits<double>::max();
  max_time = 0;
  avg_time = 0;
  n_tests  = 100;

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      Timer time;
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("system_matrix_" + std::to_string(degree)).c_str());
#  endif
      src.update_ghost_values();

      system_matrix.vmult(dst3, src);

      src.zero_out_ghost_values();
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("system_matrix_" + std::to_string(degree)).c_str());
#  endif
      const double tw = time.wall_time();
      min_time        = std::min(min_time, tw);
      max_time        = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }

  constraints.distribute(dst3);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "SystemMatrix test   in " << std::setw(11) << min_time << " "
              << std::setw(11) << avg_time << " " << std::setw(11) << max_time
              << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

#if defined FE_POINT || defined FE_POINT_VEC && defined FE_EVAL
  {
    dst2 -= dst;
    const double error = dst2.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation verification: " << error << std::endl;
  }
#endif
#if defined FE_VAL && defined FE_EVAL
  {
    dst3 -= dst;
    const double error = dst3.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEValues          verification: " << error << std::endl
                << std::endl;
  }
#endif
#if defined SYSTEM_MATRIX && defined FE_EVAL
  {
    dst3 -= dst;
    const double error = dst3.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "SystemMatrix      verification: " << error << std::endl
                << std::endl;
  }
#endif
}



template <typename Integrator, typename Number2>
void
do_flux_term(Integrator &       evaluator_m,
             Integrator &       evaluator_p,
             const Number2 &    tau,
             const unsigned int q)
{
  const auto normal_gradient_m = evaluator_m.get_normal_derivative(q);
  const auto normal_gradient_p = evaluator_p.get_normal_derivative(q);

  const auto value_m = evaluator_m.get_value(q);
  const auto value_p = evaluator_p.get_value(q);

  const auto jump_value = value_m - value_p;

  const auto central_flux_gradient =
    0.5 * (normal_gradient_m + normal_gradient_p);

  const auto value_terms = central_flux_gradient - tau * jump_value;

  evaluator_m.submit_value(-value_terms, q);
  evaluator_p.submit_value(value_terms, q);

  const auto gradient_terms = -0.5 * jump_value;

  evaluator_m.submit_normal_derivative(gradient_terms, q);
  evaluator_p.submit_normal_derivative(gradient_terms, q);
}



template <int dim>
void
test_dg_fcl(const unsigned int degree, const unsigned int n_dofs)
{
  using namespace dealii;

  const unsigned int n_q_points = degree + 1;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  // calculate subdivisions/refinements
  const unsigned int n_cells = n_dofs / Utilities::fixed_power<dim>(degree + 1);

  const unsigned int child_cells_per_cell =
    ReferenceCells::get_hypercube<dim>().n_isotropic_children();

  unsigned int n_global_refinements = 0;
  unsigned int n_subdivisions       = 0;
  double       cells_on_coarse_grid = n_cells;
  while (cells_on_coarse_grid > 8000)
    {
      cells_on_coarse_grid /= child_cells_per_cell;
      n_global_refinements++;
    }

  if (dim == 2)
    n_subdivisions = std::ceil(std::sqrt(cells_on_coarse_grid));
  else if (dim == 3)
    n_subdivisions = std::ceil(std::cbrt(cells_on_coarse_grid));
  else
    AssertThrow(false, ExcNotImplemented());

  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);
  tria.refine_global(n_global_refinements);

  FE_DGQ<dim>     fe(degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  MappingQGeneric<dim> mapping(1);

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  typename MatrixFree<dim>::AdditionalData additional_data;
  additional_data.mapping_update_flags_inner_faces =
    update_values | update_gradients;
  additional_data.mapping_update_flags_boundary_faces =
    update_values | update_gradients;

  MatrixFree<dim> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, QGauss<1>(n_q_points), additional_data);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Working with " << fe.get_name() << " and "
              << dof_handler.n_dofs() << " dofs" << std::endl;

  LinearAlgebra::distributed::Vector<double> src, dst, dst2, dst3;
  matrix_free.initialize_dof_vector(src);
  for (auto &v : src)
    v = static_cast<double>(rand()) / RAND_MAX;

  matrix_free.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(dst2);
  matrix_free.initialize_dof_vector(dst3);

  unsigned int n_tests  = 100;
  double       min_time = std::numeric_limits<double>::max();
  double       max_time = 0;
  double       avg_time = 0;
#ifdef FE_EVAL
  for (unsigned int t = 0; t < n_tests; ++t)
    {
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("fe_evaluation_" + std::to_string(degree)).c_str());
#  endif
      Timer time;
      matrix_free.template loop<LinearAlgebra::distributed::Vector<double>,
                                LinearAlgebra::distributed::Vector<double>>(
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            const auto &range) {
          FEEvaluation<dim, -1> fe_eval(matrix_free);
          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              fe_eval.reinit(cell);
              fe_eval.gather_evaluate(src, EvaluationFlags::gradients);
              for (const unsigned int q : fe_eval.quadrature_point_indices())
                fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
              fe_eval.integrate_scatter(EvaluationFlags::gradients, dst);
            }
        },
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            const auto &range) {
          FEFaceEvaluation<dim, -1> fe_eval_m(matrix_free, true);
          FEFaceEvaluation<dim, -1> fe_eval_p(matrix_free, false);
          for (unsigned int face = range.first; face < range.second; ++face)
            {
              fe_eval_m.reinit(face);
              fe_eval_p.reinit(face);
              fe_eval_m.gather_evaluate(src,
                                        EvaluationFlags::values |
                                          EvaluationFlags::gradients);
              fe_eval_p.gather_evaluate(src,
                                        EvaluationFlags::values |
                                          EvaluationFlags::gradients);
              for (unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
                do_flux_term(fe_eval_m, fe_eval_p, 1.0, q);
              fe_eval_m.integrate_scatter(EvaluationFlags::values |
                                            EvaluationFlags::gradients,
                                          dst);
              fe_eval_p.integrate_scatter(EvaluationFlags::values |
                                            EvaluationFlags::gradients,
                                          dst);
            }
        },
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            const auto &range) {
          FEFaceEvaluation<dim, -1> fe_eval_m(matrix_free, true);
          for (unsigned int face = range.first; face < range.second; ++face)
            {
              fe_eval_m.reinit(face);
              fe_eval_m.gather_evaluate(src,
                                        EvaluationFlags::values |
                                          EvaluationFlags::gradients);
              for (unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
                {
                  const auto value    = fe_eval_m.get_value(q);
                  const auto gradient = fe_eval_m.get_gradient(q);

                  fe_eval_m.submit_value(gradient * fe_eval_m.normal_vector(q) +
                                           value,
                                         q);
                  fe_eval_m.submit_gradient(value * fe_eval_m.normal_vector(q),
                                            q);
                }
              fe_eval_m.integrate_scatter(EvaluationFlags::values |
                                            EvaluationFlags::gradients,
                                          dst);
            }
        },
        dst,
        src,
        true);
      const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("fe_evaluation_" + std::to_string(degree)).c_str());
#  endif
      min_time = std::min(min_time, tw);
      max_time = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "FEEvaluation test      in " << std::setw(11) << min_time
              << " " << std::setw(11) << avg_time << " " << std::setw(11)
              << max_time << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

  QGauss<dim>                  quad_cell(n_q_points);
  std::vector<Quadrature<dim>> quad_vec_cells;
  quad_vec_cells.reserve(
    (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
    n_lanes);

#if SURFACE_TERMS
  QGauss<dim - 1>         quad_dim_m1(degree + 1);
  std::vector<Point<dim>> points(quad_dim_m1.size());
  for (unsigned int i = 0; i < quad_dim_m1.size(); ++i)
    {
      for (unsigned int d = 0; d < dim - 1; ++d)
        points[i][d] = quad_dim_m1.get_points()[i][d];
      points[i][dim - 1] = 0.5;
    }

  std::vector<Tensor<1, dim>> normals(quad_dim_m1.size());
  for (auto &n : normals)
    n[dim - 1] = 1.;
  NonMatching::ImmersedSurfaceQuadrature<dim> quad_surface(
    points, quad_dim_m1.get_weights(), normals);
  std::vector<NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
  quad_vec_surface.reserve(matrix_free.n_cell_batches() * n_lanes);
#endif

  std::vector<typename DoFHandler<dim>::cell_iterator> vector_accessors;
  vector_accessors.reserve(
    (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
    n_lanes);
  for (unsigned int cell_batch = 0;
       cell_batch <
       matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
       ++cell_batch)
    for (unsigned int v = 0; v < n_lanes; ++v)
      {
        if (v < matrix_free.n_active_entries_per_cell_batch(cell_batch))
          vector_accessors.push_back(
            matrix_free.get_cell_iterator(cell_batch, v));
        else
          vector_accessors.push_back(
            matrix_free.get_cell_iterator(cell_batch, 0));

        quad_vec_cells.push_back(quad_cell);

#if SURFACE_TERMS
        quad_vec_surface.push_back(quad_surface);
#endif
      }

  QGauss<dim - 1>                  quad_face(n_q_points);
  std::vector<Quadrature<dim - 1>> quad_vec_faces;
  quad_vec_faces.reserve((matrix_free.n_inner_face_batches() +
                          matrix_free.n_boundary_face_batches()) *
                         n_lanes);
  std::vector<std::pair<typename DoFHandler<dim>::cell_iterator, unsigned int>>
    vector_face_accessors_m;
  vector_face_accessors_m.reserve((matrix_free.n_inner_face_batches() +
                                   matrix_free.n_boundary_face_batches()) *
                                  n_lanes);
  // fill container for inner face batches
  unsigned int face_batch = 0;
  for (; face_batch < matrix_free.n_inner_face_batches(); ++face_batch)
    {
      for (unsigned int v = 0; v < n_lanes; ++v)
        {
          if (v < matrix_free.n_active_entries_per_face_batch(face_batch))
            vector_face_accessors_m.push_back(
              matrix_free.get_face_iterator(face_batch, v, true));
          else
            vector_face_accessors_m.push_back(
              matrix_free.get_face_iterator(face_batch, 0, true));

          quad_vec_faces.push_back(quad_face);
        }
    }
  // and boundary face batches
  for (; face_batch < (matrix_free.n_inner_face_batches() +
                       matrix_free.n_boundary_face_batches());
       ++face_batch)
    {
      for (unsigned int v = 0; v < n_lanes; ++v)
        {
          if (v < matrix_free.n_active_entries_per_face_batch(face_batch))
            vector_face_accessors_m.push_back(
              matrix_free.get_face_iterator(face_batch, v, true));
          else
            vector_face_accessors_m.push_back(
              matrix_free.get_face_iterator(face_batch, 0, true));

          quad_vec_faces.push_back(quad_face);
        }
    }

#ifdef FE_POINT
  {
    min_time = std::numeric_limits<double>::max();
    max_time = 0;
    avg_time = 0;
    n_tests  = 100;

    NonMatching::MappingInfo<dim> mapping_info_cells(mapping,
                                                     update_gradients |
                                                       update_JxW_values);
    NonMatching::MappingInfo<dim> mapping_info_faces(mapping,
                                                     update_values |
                                                       update_gradients |
                                                       update_JxW_values |
                                                       update_normal_vectors);

    mapping_info_cells.reinit_cells(vector_accessors, quad_vec_cells);
    mapping_info_faces.reinit_faces(vector_face_accessors_m, quad_vec_faces);

    FEPointEvaluation<1, dim, dim, double>     fe_peval(mapping_info_cells, fe);
    FEFacePointEvaluation<1, dim, dim, double> fe_peval_m(mapping_info_faces,
                                                          fe,
                                                          true);
    FEFacePointEvaluation<1, dim, dim, double> fe_peval_p(mapping_info_faces,
                                                          fe,
                                                          false);

#  if SURFACE_TERMS
    NonMatching::MappingInfo<dim> mapping_info_surface(mapping,
                                                       update_values |
                                                         update_gradients |
                                                         update_JxW_values |
                                                         update_normal_vectors);

    mapping_info_surface.reinit_surface(vector_accessors, quad_vec_surface);

    FEPointEvaluation<1, dim, dim, double> fe_peval_surface(
      mapping_info_surface, fe);
#  endif

    for (unsigned int t = 0; t < n_tests; ++t)
      {
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_START(("fe_point_" + std::to_string(degree)).c_str());
#  endif
        Timer time;
        matrix_free.template loop<LinearAlgebra::distributed::Vector<double>,
                                  LinearAlgebra::distributed::Vector<double>>(
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEEvaluation<dim, -1> fe_eval(matrix_free);
            for (unsigned int cell = range.first; cell < range.second; ++cell)
              {
                fe_eval.reinit(cell);
                fe_eval.read_dof_values(src);
                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    fe_peval.reinit(cell * n_lanes + v);
                    fe_peval.evaluate(StridedArrayView<const double, n_lanes>(
                                        &fe_eval.begin_dof_values()[0][v],
                                        fe.dofs_per_cell),
                                      EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval.quadrature_point_indices())
                      fe_peval.submit_gradient(fe_peval.get_gradient(q), q);
                    fe_peval.integrate(StridedArrayView<double, n_lanes>(
                                         &fe_eval.begin_dof_values()[0][v],
                                         fe.dofs_per_cell),
                                       EvaluationFlags::gradients);

#  if SURFACE_TERMS
                    fe_peval_surface.reinit(cell * n_lanes + v);
                    fe_peval_surface.evaluate(
                      StridedArrayView<const double, n_lanes>(
                        &fe_eval.begin_dof_values()[0][v],
                        fe_eval.dofs_per_cell),
                      EvaluationFlags::gradients | EvaluationFlags::values);
                    for (const unsigned int q :
                         fe_peval_surface.quadrature_point_indices())
                      {
                        const auto &value    = fe_peval_surface.get_value(q);
                        const auto &gradient = fe_peval_surface.get_gradient(q);

                        fe_peval_surface.submit_value(
                          fe_peval_surface.normal_vector(q) * gradient * 0., q);
                        fe_peval_surface.submit_gradient(
                          fe_peval_surface.normal_vector(q) * value * 0., q);
                      }
                    fe_peval_surface.integrate(
                      StridedArrayView<double, n_lanes>(
                        &fe_eval.begin_dof_values()[0][v],
                        fe_eval.dofs_per_cell),
                      EvaluationFlags::gradients | EvaluationFlags::values,
                      true);
#  endif
                  }
                fe_eval.distribute_local_to_global(dst);
              }
          },
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEFaceEvaluation<dim, -1> fe_eval_m(matrix_free, true);
            FEFaceEvaluation<dim, -1> fe_eval_p(matrix_free, false);
            for (unsigned int face = range.first; face < range.second; ++face)
              {
                fe_eval_m.reinit(face);
                fe_eval_p.reinit(face);

                fe_eval_m.read_dof_values(src);
                fe_eval_p.read_dof_values(src);

                fe_eval_m.project_to_face(EvaluationFlags::values |
                                          EvaluationFlags::gradients);
                fe_eval_p.project_to_face(EvaluationFlags::values |
                                          EvaluationFlags::gradients);

                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    fe_peval_m.reinit(face * n_lanes + v);
                    fe_peval_p.reinit(face * n_lanes + v);
                    fe_peval_m.evaluate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    fe_peval_p.evaluate_in_face(
                      &fe_eval_p.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval_m.quadrature_point_indices())
                      do_flux_term(fe_peval_m, fe_peval_p, 1.0, q);
                    fe_peval_m.integrate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    fe_peval_p.integrate_in_face(
                      &fe_eval_p.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                  }

                fe_eval_m.collect_from_face(EvaluationFlags::values |
                                            EvaluationFlags::gradients);
                fe_eval_p.collect_from_face(EvaluationFlags::values |
                                            EvaluationFlags::gradients);

                fe_eval_m.distribute_local_to_global(dst);
                fe_eval_p.distribute_local_to_global(dst);
              }
          },
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEFaceEvaluation<dim, -1> fe_eval_m(matrix_free, true);
            for (unsigned int face = range.first; face < range.second; ++face)
              {
                fe_eval_m.reinit(face);

                fe_eval_m.read_dof_values(src);

                fe_eval_m.project_to_face(EvaluationFlags::values |
                                          EvaluationFlags::gradients);

                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    fe_peval_m.reinit(face * n_lanes + v);
                    fe_peval_m.evaluate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval_m.quadrature_point_indices())
                      {
                        const auto value    = fe_peval_m.get_value(q);
                        const auto gradient = fe_peval_m.get_gradient(q);

                        fe_peval_m.submit_value(
                          gradient * fe_peval_m.normal_vector(q) + value, q);
                        fe_peval_m.submit_gradient(
                          value * fe_peval_m.normal_vector(q), q);
                      }
                    fe_peval_m.integrate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                  }

                fe_eval_m.collect_from_face(EvaluationFlags::values |
                                            EvaluationFlags::gradients);

                fe_eval_m.distribute_local_to_global(dst);
              }
          },
          dst2,
          src,
          true);
        const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP(("fe_point_" + std::to_string(degree)).c_str());
#  endif
        min_time = std::min(min_time, tw);
        max_time = std::max(max_time, tw);
        avg_time += tw / n_tests;
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation test in " << std::setw(11) << min_time
                << " " << std::setw(11) << avg_time << " " << std::setw(11)
                << max_time << " seconds, throughput " << std::setw(8)
                << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
                << std::endl;
  }
#endif

#ifdef FE_POINT_VEC
  {
    min_time = std::numeric_limits<double>::max();
    max_time = 0;
    avg_time = 0;
    n_tests  = 100;

    NonMatching::MappingInfo<dim, dim, VectorizedArray<double>>
      mapping_info_cells_vec(mapping, update_gradients | update_JxW_values);
    NonMatching::MappingInfo<dim, dim, VectorizedArray<double>>
      mapping_info_faces_vec(mapping,
                             update_values | update_gradients |
                               update_JxW_values | update_normal_vectors);

    mapping_info_cells_vec.reinit_cells(vector_accessors, quad_vec_cells);
    mapping_info_faces_vec.reinit_faces(vector_face_accessors_m,
                                        quad_vec_faces);

    FEPointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval(
      mapping_info_cells_vec, fe);
    FEFacePointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval_m(
      mapping_info_faces_vec, fe);
    FEFacePointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval_p(
      mapping_info_faces_vec, fe);

#  if SURFACE_TERMS
    NonMatching::MappingInfo<dim, dim, VectorizedArray<double>>
      mapping_info_surface_vec(mapping,
                               update_values | update_gradients |
                                 update_JxW_values | update_normal_vectors);

    mapping_info_surface_vec.reinit_surface(vector_accessors, quad_vec_surface);

    FEPointEvaluation<1, dim, dim, VectorizedArray<double>>
      fe_peval_surface_vec(mapping_info_surface_vec, fe);
#  endif

    for (unsigned int t = 0; t < n_tests; ++t)
      {
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_START(("fe_point_vec_" + std::to_string(degree)).c_str());
#  endif
        Timer time;
        matrix_free.template loop<LinearAlgebra::distributed::Vector<double>,
                                  LinearAlgebra::distributed::Vector<double>>(
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEEvaluation<dim, -1> fe_eval(matrix_free);
            for (unsigned int cell = range.first; cell < range.second; ++cell)
              {
                fe_eval.reinit(cell);
                fe_eval.read_dof_values(src);
                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    fe_peval.reinit(cell * n_lanes + v);
                    fe_peval.evaluate(StridedArrayView<const double, n_lanes>(
                                        &fe_eval.begin_dof_values()[0][v],
                                        fe.dofs_per_cell),
                                      EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval.quadrature_point_indices())
                      fe_peval.submit_gradient(fe_peval.get_gradient(q), q);
                    fe_peval.integrate(StridedArrayView<double, n_lanes>(
                                         &fe_eval.begin_dof_values()[0][v],
                                         fe.dofs_per_cell),
                                       EvaluationFlags::gradients);

#  if SURFACE_TERMS
                    fe_peval_surface_vec.reinit(cell * n_lanes + v);
                    fe_peval_surface_vec.evaluate(
                      StridedArrayView<const double, n_lanes>(
                        &fe_eval.begin_dof_values()[0][v],
                        fe_eval.dofs_per_cell),
                      EvaluationFlags::gradients | EvaluationFlags::values);
                    for (const unsigned int q :
                         fe_peval_surface_vec.quadrature_point_indices())
                      {
                        const auto &value = fe_peval_surface_vec.get_value(q);
                        const auto &gradient =
                          fe_peval_surface_vec.get_gradient(q);

                        fe_peval_surface_vec.submit_value(
                          fe_peval_surface_vec.normal_vector(q) * gradient * 0.,
                          q);
                        fe_peval_surface_vec.submit_gradient(
                          fe_peval_surface_vec.normal_vector(q) * value * 0.,
                          q);
                      }
                    fe_peval_surface_vec.integrate(
                      StridedArrayView<double, n_lanes>(
                        &fe_eval.begin_dof_values()[0][v],
                        fe_eval.dofs_per_cell),
                      EvaluationFlags::gradients | EvaluationFlags::values,
                      true);
#  endif
                  }
                fe_eval.distribute_local_to_global(dst);
              }
          },
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEFaceEvaluation<dim, -1> fe_eval_m(matrix_free, true);
            FEFaceEvaluation<dim, -1> fe_eval_p(matrix_free, false);
            for (unsigned int face = range.first; face < range.second; ++face)
              {
                fe_eval_m.reinit(face);
                fe_eval_p.reinit(face);

                fe_eval_m.read_dof_values(src);
                fe_eval_p.read_dof_values(src);

                fe_eval_m.project_to_face(EvaluationFlags::values |
                                          EvaluationFlags::gradients);
                fe_eval_p.project_to_face(EvaluationFlags::values |
                                          EvaluationFlags::gradients);

                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    fe_peval_m.reinit(face * n_lanes + v);
                    fe_peval_p.reinit(face * n_lanes + v);
                    fe_peval_m.evaluate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    fe_peval_p.evaluate_in_face(
                      &fe_eval_p.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval_m.quadrature_point_indices())
                      do_flux_term(fe_peval_m, fe_peval_p, 1.0, q);
                    fe_peval_m.integrate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    fe_peval_p.integrate_in_face(
                      &fe_eval_p.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                  }

                fe_eval_m.collect_from_face(EvaluationFlags::values |
                                            EvaluationFlags::gradients);
                fe_eval_p.collect_from_face(EvaluationFlags::values |
                                            EvaluationFlags::gradients);

                fe_eval_m.distribute_local_to_global(dst);
                fe_eval_p.distribute_local_to_global(dst);
              }
          },
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEFaceEvaluation<dim, -1> fe_eval_m(matrix_free, true);
            for (unsigned int face = range.first; face < range.second; ++face)
              {
                fe_eval_m.reinit(face);

                fe_eval_m.read_dof_values(src);

                fe_eval_m.project_to_face(EvaluationFlags::values |
                                          EvaluationFlags::gradients);

                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    fe_peval_m.reinit(face * n_lanes + v);
                    fe_peval_m.evaluate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval_m.quadrature_point_indices())
                      {
                        const auto value    = fe_peval_m.get_value(q);
                        const auto gradient = fe_peval_m.get_gradient(q);

                        fe_peval_m.submit_value(
                          gradient * fe_peval_m.normal_vector(q) + value, q);
                        fe_peval_m.submit_gradient(
                          value * fe_peval_m.normal_vector(q), q);
                      }
                    fe_peval_m.integrate_in_face(
                      &fe_eval_m.get_scratch_data().begin()[0][v],
                      EvaluationFlags::values | EvaluationFlags::gradients);
                  }

                fe_eval_m.collect_from_face(EvaluationFlags::values |
                                            EvaluationFlags::gradients);

                fe_eval_m.distribute_local_to_global(dst);
              }
          },
          dst2,
          src,
          true);
        const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP(("fe_point_vec_" + std::to_string(degree)).c_str());
#  endif
        min_time = std::min(min_time, tw);
        max_time = std::max(max_time, tw);
        avg_time += tw / n_tests;
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation (vectorized) test in " << std::setw(11)
                << min_time << " " << std::setw(11) << avg_time << " "
                << std::setw(11) << max_time << " seconds, throughput "
                << std::setw(8) << 1e-6 * dof_handler.n_dofs() / avg_time
                << " MDoFs/s" << std::endl;
  }
#endif

#ifdef FE_VAL
  min_time = std::numeric_limits<double>::max();
  max_time = 0;
  avg_time = 0;
  n_tests  = 20;

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      Timer                       time;
      QGauss<dim - 1>             quad(n_q_points);
      FEFaceValues<dim>           fe_values(mapping,
                                  fe,
                                  quad,
                                  update_gradients | update_JxW_values);
      Vector<double>              solution_values_in(fe.dofs_per_cell);
      Vector<double>              solution_values_out(fe.dofs_per_cell);
      Vector<double>              solution_values_sum(fe.dofs_per_cell);
      std::vector<Tensor<1, dim>> solution_gradients(quad.size());
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      src.update_ghost_values();
      dst3 = 0;

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(dof_indices);
            constraints.get_dof_values(src,
                                       dof_indices.begin(),
                                       solution_values_in.begin(),
                                       solution_values_in.end());

            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              solution_values_sum[i] = 0.;

            for (const auto f : cell->face_indices())
              {
                fe_values.reinit(cell, f);
                for (unsigned int q = 0; q < quad.size(); ++q)
                  {
                    Tensor<1, dim> gradient;
                    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                      gradient +=
                        solution_values_in(i) * fe_values.shape_grad(i, q);
                    solution_gradients[q] = gradient * fe_values.JxW(q);
                  }
                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  {
                    double sum = 0;
                    for (unsigned int q = 0; q < quad.size(); ++q)
                      sum += solution_gradients[q] * fe_values.shape_grad(i, q);
                    solution_values_out(i) = sum;
                  }

                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  solution_values_sum[i] += solution_values_out[i];
              }

            constraints.distribute_local_to_global(solution_values_sum,
                                                   dof_indices,
                                                   dst3);
          }
      dst3.compress(VectorOperation::add);
      src.zero_out_ghost_values();
      const double tw = time.wall_time();
      min_time        = std::min(min_time, tw);
      max_time        = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "FEValues test          in " << std::setw(11) << min_time
              << " " << std::setw(11) << avg_time << " " << std::setw(11)
              << max_time << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

#ifdef SYSTEM_MATRIX
  const auto locally_owned_dofs = dof_handler.locally_owned_dofs();
  const auto locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);
  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);

  TrilinosWrappers::SparseMatrix system_matrix;

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       MPI_COMM_WORLD);

  FEValues<dim>     fe_values(fe,
                          quad_cell,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values_m(fe,
                                     quad_face,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);
  FEFaceValues<dim> fe_face_values_p(fe,
                                     quad_face,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> face_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_p(dofs_per_cell);

  Timer time;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0.;

        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);

        for (const unsigned int f : cell->face_indices())
          {
            fe_face_values_m.reinit(cell, f);

            if (cell->at_boundary(f))
              {
                for (unsigned int q = 0;
                     q < fe_face_values_m.n_quadrature_points;
                     ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      cell_matrix(i, j) +=
                        (fe_face_values_m.shape_grad(i, q) *
                           fe_face_values_m.shape_value(j, q) *
                           fe_face_values_m.normal_vector(q) +
                         fe_face_values_m.shape_value(i, q) *
                           (fe_face_values_m.shape_grad(j, q) *
                              fe_face_values_m.normal_vector(q) +
                            fe_face_values_m.shape_value(j, q))) *
                        fe_face_values_m.JxW(q);
              }
            else
              {
                for (unsigned int q = 0;
                     q < fe_face_values_m.n_quadrature_points;
                     ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      cell_matrix(i, j) +=
                        (-fe_face_values_m.shape_grad(i, q) * 0.5 *
                           fe_face_values_m.shape_value(j, q) *
                           fe_face_values_m.normal_vector(q) +
                         -fe_face_values_m.shape_value(i, q) *
                           (0.5 * fe_face_values_m.shape_grad(j, q) -
                            fe_face_values_m.shape_value(j, q) *
                              fe_face_values_m.normal_vector(q)) *
                           fe_face_values_m.normal_vector(q)) *
                        fe_face_values_m.JxW(q);

                fe_face_values_p.reinit(cell->neighbor(f),
                                        cell->neighbor_of_neighbor(f));

                face_matrix = 0.;

                for (unsigned int q = 0;
                     q < fe_face_values_m.n_quadrature_points;
                     ++q)
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      face_matrix(i, j) +=
                        (fe_face_values_m.shape_grad(i, q) * 0.5 *
                           fe_face_values_p.shape_value(j, q) *
                           fe_face_values_m.normal_vector(q) +
                         fe_face_values_m.shape_value(i, q) *
                           -(0.5 * fe_face_values_p.shape_grad(j, q) +
                             fe_face_values_p.shape_value(j, q) *
                               fe_face_values_m.normal_vector(q)) *
                           fe_face_values_m.normal_vector(q)) *
                        fe_face_values_m.JxW(q);

                cell->neighbor(f)->get_dof_indices(local_dof_indices_p);
                constraints.distribute_local_to_global(face_matrix,
                                                       local_dof_indices,
                                                       local_dof_indices_p,
                                                       system_matrix);
              }
          }

        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix);
      }

  system_matrix.compress(VectorOperation::add);
  const double tw = time.wall_time();

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Assembly time: " << tw << std::endl;

  min_time = std::numeric_limits<double>::max();
  max_time = 0;
  avg_time = 0;
  n_tests  = 100;

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      Timer time;
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("system_matrix_" + std::to_string(degree)).c_str());
#  endif
      src.update_ghost_values();

      system_matrix.vmult(dst3, src);

      src.zero_out_ghost_values();
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("system_matrix_" + std::to_string(degree)).c_str());
#  endif
      const double tw = time.wall_time();
      min_time        = std::min(min_time, tw);
      max_time        = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }

  constraints.distribute(dst3);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "SystemMatrix test   in " << std::setw(11) << min_time << " "
              << std::setw(11) << avg_time << " " << std::setw(11) << max_time
              << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

#if defined FE_POINT || defined FE_POINT_VEC && defined FE_EVAL
  {
    dst2 -= dst;
    const double error = dst2.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation verification: " << error << std::endl;
  }
#endif
#if defined FE_VAL && defined FE_EVAL
  {
    dst3 -= dst;
    const double error = dst3.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEValues          verification: " << error << std::endl
                << std::endl;
  }
#endif
#if defined SYSTEM_MATRIX && defined FE_EVAL
  {
    dst3 -= dst;
    const double error = dst3.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "SystemMatrix      verification: " << error << std::endl
                << std::endl;
  }
#endif
}



template <typename Integrator, typename Number2>
void
do_flux_term_ecl(Integrator &       evaluator_m,
                 Integrator &       evaluator_p,
                 const Number2 &    tau,
                 const unsigned int q)
{
  const auto gradient_m = evaluator_m.get_gradient(q);
  const auto gradient_p = evaluator_p.get_gradient(q);

  const auto value_m = evaluator_m.get_value(q);
  const auto value_p = evaluator_p.get_value(q);

  const auto normal = evaluator_m.normal_vector(q);

  const auto jump_value = (value_m - value_p) * normal;

  const auto central_flux_gradient = 0.5 * (gradient_m + gradient_p);

  const auto value_terms = normal * (central_flux_gradient - tau * jump_value);

  evaluator_m.submit_value(-value_terms, q);

  evaluator_m.submit_gradient(-0.5 * jump_value, q);
}



template <int dim>
void
test_dg_ecl(const unsigned int degree, const unsigned int n_dofs)
{
  using namespace dealii;

  const unsigned int n_q_points = degree + 1;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  // calculate subdivisions/refinements
  const unsigned int n_cells = n_dofs / Utilities::fixed_power<dim>(degree + 1);

  const unsigned int child_cells_per_cell =
    ReferenceCells::get_hypercube<dim>().n_isotropic_children();

  unsigned int n_global_refinements = 0;
  unsigned int n_subdivisions       = 0;
  double       cells_on_coarse_grid = n_cells;
  while (cells_on_coarse_grid > 8000)
    {
      cells_on_coarse_grid /= child_cells_per_cell;
      n_global_refinements++;
    }

  if (dim == 2)
    n_subdivisions = std::ceil(std::sqrt(cells_on_coarse_grid));
  else if (dim == 3)
    n_subdivisions = std::ceil(std::cbrt(cells_on_coarse_grid));
  else
    AssertThrow(false, ExcNotImplemented());

  GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);
  tria.refine_global(n_global_refinements);

  FE_DGQ<dim>     fe(degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  MappingQGeneric<dim> mapping(1);

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  typename MatrixFree<dim>::AdditionalData additional_data;
  additional_data.mapping_update_flags_inner_faces =
    update_values | update_gradients;
  additional_data.mapping_update_flags_boundary_faces =
    update_values | update_gradients;
  additional_data.hold_all_faces_to_owned_cells = true;
  additional_data.mapping_update_flags_faces_by_cells =
    additional_data.mapping_update_flags_inner_faces |
    additional_data.mapping_update_flags_boundary_faces;

  MatrixFree<dim> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, QGauss<1>(n_q_points), additional_data);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Working with " << fe.get_name() << " and "
              << dof_handler.n_dofs() << " dofs" << std::endl;

  LinearAlgebra::distributed::Vector<double> src, dst, dst2, dst3;
  matrix_free.initialize_dof_vector(src);
  for (auto &v : src)
    v = static_cast<double>(rand()) / RAND_MAX;

  matrix_free.initialize_dof_vector(dst);
  matrix_free.initialize_dof_vector(dst2);
  matrix_free.initialize_dof_vector(dst3);

  unsigned int n_tests  = 100;
  double       min_time = std::numeric_limits<double>::max();
  double       max_time = 0;
  double       avg_time = 0;
#ifdef FE_EVAL
  for (unsigned int t = 0; t < n_tests; ++t)
    {
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("fe_evaluation_" + std::to_string(degree)).c_str());
#  endif
      Timer time;
      matrix_free
        .template loop_cell_centric<LinearAlgebra::distributed::Vector<double>,
                                    LinearAlgebra::distributed::Vector<double>>(
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEEvaluation<dim, -1>     fe_eval(matrix_free);
            FEFaceEvaluation<dim, -1> fe_eval_m(matrix_free, true);
            FEFaceEvaluation<dim, -1> fe_eval_p(matrix_free, false);
            AlignedVector<VectorizedArray<double>> vec_solution_values_in_m(
              fe_eval.dofs_per_cell);
            for (unsigned int cell = range.first; cell < range.second; ++cell)
              {
                fe_eval.reinit(cell);
                fe_eval.read_dof_values(src);
                for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
                  vec_solution_values_in_m[i] = fe_eval.begin_dof_values()[i];
                fe_eval.evaluate(EvaluationFlags::gradients);

                for (const unsigned int q : fe_eval.quadrature_point_indices())
                  fe_eval.submit_gradient(fe_eval.get_gradient(q), q);

                fe_eval.integrate(EvaluationFlags::gradients);

                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                     ++f)
                  {
                    // ask for boundary ids of face
                    const auto boundary_ids =
                      matrix_free.get_faces_by_cells_boundary_id(cell, f);

                    // only internal faces have a neighbor, setup a mask
                    std::bitset<n_lanes>    mask;
                    VectorizedArray<double> fluxes = 0.;
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      {
                        mask[v] =
                          boundary_ids[v] == numbers::internal_face_boundary_id;
                        fluxes[v] = mask[v] == true ? 1. : 0.;
                      }

                    fe_eval_m.reinit(cell, f);
                    fe_eval_p.reinit(cell, f);

                    fe_eval_p.read_dof_values(src, 0, mask);

                    fe_eval_m.evaluate(vec_solution_values_in_m.data(),
                                       EvaluationFlags::values |
                                         EvaluationFlags::gradients);
                    fe_eval_p.evaluate(EvaluationFlags::values |
                                       EvaluationFlags::gradients);

                    for (const auto q : fe_eval_m.quadrature_point_indices())
                      do_flux_term_ecl(fe_eval_m, fe_eval_p, 1.0, q);

                    fe_eval_m.integrate(EvaluationFlags::values |
                                        EvaluationFlags::gradients);

                    for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
                      fe_eval.begin_dof_values()[i] +=
                        fe_eval_m.begin_dof_values()[i] * fluxes;
                  }

                fe_eval.distribute_local_to_global(dst);
              }
          },
          dst,
          src,
          true);
      const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("fe_evaluation_" + std::to_string(degree)).c_str());
#  endif
      min_time = std::min(min_time, tw);
      max_time = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "FEEvaluation test      in " << std::setw(11) << min_time
              << " " << std::setw(11) << avg_time << " " << std::setw(11)
              << max_time << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

  QGauss<dim>                  quad_cell(n_q_points);
  QGauss<dim - 1>              quad_face(n_q_points);
  std::vector<Quadrature<dim>> quad_vec_cells;
  quad_vec_cells.reserve(
    (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
    n_lanes);
  std::vector<std::vector<Quadrature<dim - 1>>> quad_vec_faces(
    (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
    n_lanes);

  std::vector<typename DoFHandler<dim>::cell_iterator> vector_accessors;
  vector_accessors.reserve(
    (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
    n_lanes);
  for (unsigned int cell_batch = 0;
       cell_batch <
       matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
       ++cell_batch)
    for (unsigned int v = 0; v < n_lanes; ++v)
      {
        if (v < matrix_free.n_active_entries_per_cell_batch(cell_batch))
          vector_accessors.push_back(
            matrix_free.get_cell_iterator(cell_batch, v));
        else
          vector_accessors.push_back(
            matrix_free.get_cell_iterator(cell_batch, 0));

        quad_vec_cells.push_back(quad_cell);

        for (const auto f : GeometryInfo<dim>::face_indices())
          {
            (void)f;
            quad_vec_faces[cell_batch * n_lanes + v].push_back(quad_face);
          }
      }

#ifdef FE_POINT
  {
    min_time = std::numeric_limits<double>::max();
    max_time = 0;
    avg_time = 0;
    n_tests  = 100;

    NonMatching::MappingInfo<dim> mapping_info_cells(mapping,
                                                     update_gradients |
                                                       update_JxW_values);
    NonMatching::MappingInfo<dim> mapping_info_faces(mapping,
                                                     update_values |
                                                       update_gradients |
                                                       update_JxW_values |
                                                       update_normal_vectors);

    mapping_info_cells.reinit_cells(vector_accessors, quad_vec_cells);
    mapping_info_faces.reinit_faces(vector_accessors, quad_vec_faces);

    FEPointEvaluation<1, dim, dim, double>     fe_peval(mapping_info_cells, fe);
    FEFacePointEvaluation<1, dim, dim, double> fe_peval_m(mapping_info_faces,
                                                          fe);
    FEFacePointEvaluation<1, dim, dim, double> fe_peval_p(mapping_info_faces,
                                                          fe);

    for (unsigned int t = 0; t < n_tests; ++t)
      {
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_START(("fe_point_" + std::to_string(degree)).c_str());
#  endif
        Timer time;
        matrix_free.template loop_cell_centric<
          LinearAlgebra::distributed::Vector<double>,
          LinearAlgebra::distributed::Vector<double>>(
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEEvaluation<dim, -1>     fe_eval(matrix_free);
            FEFaceEvaluation<dim, -1> fe_eval_p(matrix_free, false);
            AlignedVector<VectorizedArray<double>> vec_solution_values_in_m(
              fe_eval.dofs_per_cell);
            for (unsigned int cell = range.first; cell < range.second; ++cell)
              {
                fe_eval.reinit(cell);
                fe_eval.read_dof_values(src);

                for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
                  vec_solution_values_in_m[i] = fe_eval.begin_dof_values()[i];

                for (unsigned int v = 0;
                     v < matrix_free.n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    fe_peval.reinit(cell * n_lanes + v);
                    fe_peval.evaluate(StridedArrayView<const double, n_lanes>(
                                        &vec_solution_values_in_m[0][v],
                                        fe.dofs_per_cell),
                                      EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval.quadrature_point_indices())
                      fe_peval.submit_gradient(fe_peval.get_gradient(q), q);
                    fe_peval.integrate(StridedArrayView<double, n_lanes>(
                                         &fe_eval.begin_dof_values()[0][v],
                                         fe.dofs_per_cell),
                                       EvaluationFlags::gradients);
                  }

                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                     ++f)
                  {
                    // ask for boundary ids of face
                    const auto boundary_ids =
                      matrix_free.get_faces_by_cells_boundary_id(cell, f);

                    // only internal faces have a neighbor, setup a mask
                    std::bitset<n_lanes> mask;
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      mask[v] =
                        boundary_ids[v] == numbers::internal_face_boundary_id;

                    fe_eval_p.reinit(cell, f);
                    fe_eval_p.read_dof_values(src, 0, mask);

                    for (unsigned int v = 0;
                         v < matrix_free.n_active_entries_per_cell_batch(cell);
                         ++v)
                      {
                        if (mask[v] == false)
                          continue;

                        fe_peval_m.reinit(cell * n_lanes + v, f);

                        const auto &cell_iterator =
                          matrix_free.get_cell_iterator(cell, v);

                        fe_peval_p.reinit(
                          matrix_free.get_matrix_free_cell_index(
                            cell_iterator->neighbor(f)),
                          cell_iterator->neighbor_of_neighbor(f));

                        fe_peval_m.evaluate(
                          StridedArrayView<const double, n_lanes>(
                            &vec_solution_values_in_m[0][v], fe.dofs_per_cell),
                          EvaluationFlags::values | EvaluationFlags::gradients);
                        fe_peval_p.evaluate(
                          StridedArrayView<const double, n_lanes>(
                            &fe_eval_p.begin_dof_values()[0][v],
                            fe.dofs_per_cell),
                          EvaluationFlags::values | EvaluationFlags::gradients);

                        for (const unsigned int q :
                             fe_peval_m.quadrature_point_indices())
                          do_flux_term_ecl(fe_peval_m, fe_peval_p, 1.0, q);

                        fe_peval_m.integrate(
                          StridedArrayView<double, n_lanes>(
                            &fe_eval.begin_dof_values()[0][v],
                            fe.dofs_per_cell),
                          EvaluationFlags::values | EvaluationFlags::gradients,
                          true);
                      }
                  }
                fe_eval.distribute_local_to_global(dst);
              }
          },
          dst2,
          src,
          true);
        const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP(("fe_point_" + std::to_string(degree)).c_str());
#  endif
        min_time = std::min(min_time, tw);
        max_time = std::max(max_time, tw);
        avg_time += tw / n_tests;
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation test in " << std::setw(11) << min_time
                << " " << std::setw(11) << avg_time << " " << std::setw(11)
                << max_time << " seconds, throughput " << std::setw(8)
                << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
                << std::endl;
  }
#endif

#ifdef FE_POINT_VEC
  {
    min_time = std::numeric_limits<double>::max();
    max_time = 0;
    avg_time = 0;
    n_tests  = 100;

    NonMatching::MappingInfo<dim, dim, VectorizedArray<double>>
      mapping_info_cells_vec(mapping, update_gradients | update_JxW_values);
    NonMatching::MappingInfo<dim, dim, VectorizedArray<double>>
      mapping_info_faces_vec(mapping,
                             update_values | update_gradients |
                               update_JxW_values | update_normal_vectors);

    mapping_info_cells_vec.reinit_cells(vector_accessors, quad_vec_cells);
    mapping_info_faces_vec.reinit_faces(vector_accessors, quad_vec_faces);

    FEPointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval(
      mapping_info_cells_vec, fe);
    FEFacePointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval_m(
      mapping_info_faces_vec, fe);
    FEFacePointEvaluation<1, dim, dim, VectorizedArray<double>> fe_peval_p(
      mapping_info_faces_vec, fe);

    for (unsigned int t = 0; t < n_tests; ++t)
      {
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_START(("fe_point_vec_" + std::to_string(degree)).c_str());
#  endif
        Timer time;
        matrix_free.template loop_cell_centric<
          LinearAlgebra::distributed::Vector<double>,
          LinearAlgebra::distributed::Vector<double>>(
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto &range) {
            FEEvaluation<dim, -1>     fe_eval(matrix_free);
            FEFaceEvaluation<dim, -1> fe_eval_p(matrix_free, false);
            AlignedVector<VectorizedArray<double>> vec_solution_values_in_m(
              fe_eval.dofs_per_cell);
            for (unsigned int cell = range.first; cell < range.second; ++cell)
              {
                fe_eval.reinit(cell);
                fe_eval.read_dof_values(src);

                for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
                  vec_solution_values_in_m[i] = fe_eval.begin_dof_values()[i];

                for (unsigned int v = 0;
                     v < matrix_free.n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    fe_peval.reinit(cell * n_lanes + v);
                    fe_peval.evaluate(StridedArrayView<const double, n_lanes>(
                                        &vec_solution_values_in_m[0][v],
                                        fe.dofs_per_cell),
                                      EvaluationFlags::gradients);
                    for (const unsigned int q :
                         fe_peval.quadrature_point_indices())
                      fe_peval.submit_gradient(fe_peval.get_gradient(q), q);
                    fe_peval.integrate(StridedArrayView<double, n_lanes>(
                                         &fe_eval.begin_dof_values()[0][v],
                                         fe.dofs_per_cell),
                                       EvaluationFlags::gradients);
                  }

                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                     ++f)
                  {
                    // ask for boundary ids of face
                    const auto boundary_ids =
                      matrix_free.get_faces_by_cells_boundary_id(cell, f);

                    // only internal faces have a neighbor, setup a mask
                    std::bitset<n_lanes> mask;
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      mask[v] =
                        boundary_ids[v] == numbers::internal_face_boundary_id;

                    fe_eval_p.reinit(cell, f);
                    fe_eval_p.read_dof_values(src, 0, mask);

                    for (unsigned int v = 0;
                         v < matrix_free.n_active_entries_per_cell_batch(cell);
                         ++v)
                      {
                        if (mask[v] == false)
                          continue;

                        fe_peval_m.reinit(cell * n_lanes + v, f);

                        const auto &cell_iterator =
                          matrix_free.get_cell_iterator(cell, v);

                        fe_peval_p.reinit(
                          matrix_free.get_matrix_free_cell_index(
                            cell_iterator->neighbor(f)),
                          cell_iterator->neighbor_of_neighbor(f));

                        fe_peval_m.evaluate(
                          StridedArrayView<const double, n_lanes>(
                            &vec_solution_values_in_m[0][v], fe.dofs_per_cell),
                          EvaluationFlags::values | EvaluationFlags::gradients);
                        fe_peval_p.evaluate(
                          StridedArrayView<const double, n_lanes>(
                            &fe_eval_p.begin_dof_values()[0][v],
                            fe.dofs_per_cell),
                          EvaluationFlags::values | EvaluationFlags::gradients);

                        for (const unsigned int q :
                             fe_peval_m.quadrature_point_indices())
                          do_flux_term_ecl(fe_peval_m, fe_peval_p, 1.0, q);

                        fe_peval_m.integrate(
                          StridedArrayView<double, n_lanes>(
                            &fe_eval.begin_dof_values()[0][v],
                            fe.dofs_per_cell),
                          EvaluationFlags::values | EvaluationFlags::gradients,
                          true);
                      }
                  }
                fe_eval.distribute_local_to_global(dst);
              }
          },
          dst2,
          src,
          true);
        const double tw = time.wall_time();
#  ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP(("fe_point_vec_" + std::to_string(degree)).c_str());
#  endif
        min_time = std::min(min_time, tw);
        max_time = std::max(max_time, tw);
        avg_time += tw / n_tests;
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation (vectorized) test in " << std::setw(11)
                << min_time << " " << std::setw(11) << avg_time << " "
                << std::setw(11) << max_time << " seconds, throughput "
                << std::setw(8) << 1e-6 * dof_handler.n_dofs() / avg_time
                << " MDoFs/s" << std::endl;
  }
#endif

#ifdef FE_VAL
  min_time = std::numeric_limits<double>::max();
  max_time = 0;
  avg_time = 0;
  n_tests  = 20;

  for (unsigned int t = 0; t < n_tests; ++t)
    {
      Timer                       time;
      QGauss<dim - 1>             quad(n_q_points);
      FEFaceValues<dim>           fe_values(mapping,
                                  fe,
                                  quad,
                                  update_gradients | update_JxW_values);
      Vector<double>              solution_values_in(fe.dofs_per_cell);
      Vector<double>              solution_values_out(fe.dofs_per_cell);
      Vector<double>              solution_values_sum(fe.dofs_per_cell);
      std::vector<Tensor<1, dim>> solution_gradients(quad.size());
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      src.update_ghost_values();
      dst3 = 0;

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(dof_indices);
            constraints.get_dof_values(src,
                                       dof_indices.begin(),
                                       solution_values_in.begin(),
                                       solution_values_in.end());

            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              solution_values_sum[i] = 0.;

            for (const auto f : cell->face_indices())
              {
                fe_values.reinit(cell, f);
                for (unsigned int q = 0; q < quad.size(); ++q)
                  {
                    Tensor<1, dim> gradient;
                    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                      gradient +=
                        solution_values_in(i) * fe_values.shape_grad(i, q);
                    solution_gradients[q] = gradient * fe_values.JxW(q);
                  }
                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  {
                    double sum = 0;
                    for (unsigned int q = 0; q < quad.size(); ++q)
                      sum += solution_gradients[q] * fe_values.shape_grad(i, q);
                    solution_values_out(i) = sum;
                  }

                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  solution_values_sum[i] += solution_values_out[i];
              }

            constraints.distribute_local_to_global(solution_values_sum,
                                                   dof_indices,
                                                   dst3);
          }
      dst3.compress(VectorOperation::add);
      src.zero_out_ghost_values();
      const double tw = time.wall_time();
      min_time        = std::min(min_time, tw);
      max_time        = std::max(max_time, tw);
      avg_time += tw / n_tests;
    }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "FEValues test          in " << std::setw(11) << min_time
              << " " << std::setw(11) << avg_time << " " << std::setw(11)
              << max_time << " seconds, throughput " << std::setw(8)
              << 1e-6 * dof_handler.n_dofs() / avg_time << " MDoFs/s"
              << std::endl;
#endif

#if defined FE_POINT || defined FE_POINT_VEC && defined FE_EVAL
  {
    dst2 -= dst;
    const double error = dst2.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEPointEvaluation verification: " << error << std::endl;
  }
#endif
#if defined FE_VAL && defined FE_EVAL
  {
    dst3 -= dst;
    const double error = dst3.l2_norm() / dst.l2_norm();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "FEValues          verification: " << error << std::endl
                << std::endl;
  }
#endif
}



int
main(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  enum AnalysisType
  {
    instructions,
    throughput
  };

  enum BenchmarkType
  {
    cg,
    dg_fcl,
    dg_ecl
  };

  BenchmarkType benchmark_type = dg_fcl;
  AnalysisType  analysis_type  = throughput;

  const unsigned int n_dofs = 1e7;

  if (analysis_type == throughput)
    {
      if (benchmark_type == cg)
        {
          for (unsigned int i = 1; i < 8; ++i)
            test_cg<2>(i, n_dofs);
          for (unsigned int i = 1; i < 8; ++i)
            test_cg<3>(i, n_dofs);
        }
      else if (benchmark_type == dg_fcl)
        {
          for (unsigned int i = 1; i < 8; ++i)
            test_dg_fcl<2>(i, n_dofs);
          for (unsigned int i = 1; i < 8; ++i)
            test_dg_fcl<3>(i, n_dofs);
        }
      else if (benchmark_type == dg_ecl)
        {
          for (unsigned int i = 1; i < 8; ++i)
            test_dg_ecl<2>(i, n_dofs);
          for (unsigned int i = 1; i < 8; ++i)
            test_dg_ecl<3>(i, n_dofs);
        }
    }
  else if (analysis_type == instructions)
    {
      if (benchmark_type == cg)
        test_cg<3>(1, 200000);
      else if (benchmark_type == dg_fcl)
        test_dg_fcl<3>(1, 200000);
      else if (benchmark_type == dg_ecl)
        test_dg_ecl<3>(1, 200000);
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
