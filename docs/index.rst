:html_theme.sidebar_secondary.remove:
:sd_hide_title: true

SplineBox
=========

.. raw:: html

   <style>
     .h1 {
       margin-top: 1em;
       margin-bottom: 0.3em;
     }
     .h2 {
       margin: 0.5em 0 0.5em;
     }
   </style>
.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started/index.rst
   auto_examples/index.rst
   theory/index.rst
   api/index.rst

.. container:: twocol

  .. container:: col

    .. raw:: html

       <h1 style="margin-bottom: 0.2rem">SplineBox</h1>
       <h2 style="margin-top: 0.2rem">Take control of your splines</h2>
       <p style="margin-right: 3em">
         SplineBox is an open-source python package for anyone trying to fit splines.
         It offers a wide varaiety of spline types including Hermite splines and makes
         it easy to specify custom loss function to control spline properties such as
         smoothness.
       </p>

  .. container:: col

    .. code-block:: python

       import splinebox

       # number of knots
       M = 10

       basis_function = splinebox.B3()
       spline = splinebox.Spline(M, basis_function)

       spline.fit(data)

       t = np.linspace(0, M - 1, 1000)
       vals = spline.eval(t)

.. raw:: html

   <div class="grid-container" style="justify-content: left; max-width: 350px">
      <a href="./getting_started/index.html" class="button button-primary">Get Started</a>
      <a href="./auto_examples/index.html" class="button button-secondary">See Examples</a>
      <a href="./api/index.html">See API Reference →</a>
   </div>

.. raw:: html

   <h1 class="homepage-title">Key Features</h1>

   <div class="grid-container">

     <div class="grid-card">
       <svg class="w-[48px] h-[48px] text-gray-800 dark:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="none" viewBox="0 0 24 24">
         <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 9a3 3 0 0 1 3-3m-2 15h4m0-3c0-4.1 4-4.9 4-9A6 6 0 1 0 6 9c0 4 4 5 4 9h4Z"/>
       </svg>
       <p>
         <strong> Intuitive API </strong> <br/>
         Obejct oriented API design allows for easy interaction with and manipulation of splines.
       </p>
     </div>

     <div class="grid-card">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-6" width="64" height="64">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
        </svg>
       <p>
         <strong> Sensible defaults </strong> <br/>
         Automatic handling of perodicity of closed splines and padding of open splines.
         Integer values in parameter space correspond to knots.
       </p>
     </div>

     <div class="grid-card">
       <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-6" width="64" height="64">
         <path stroke-linecap="round" stroke-linejoin="round" d="M6 13.5V3.75m0 9.75a1.5 1.5 0 0 1 0 3m0-3a1.5 1.5 0 0 0 0 3m0 3.75V16.5m12-3V3.75m0 9.75a1.5 1.5 0 0 1 0 3m0-3a1.5 1.5 0 0 0 0 3m0 3.75V16.5m-6-9V3.75m0 3.75a1.5 1.5 0 0 1 0 3m0-3a1.5 1.5 0 0 0 0 3m0 9.75V10.5" />
       </svg>
       <p>
         <strong> Custom loss functions </strong> <br/>
         Compatibility with python optimization frame works makes it easy to use custom loss functions for fitting.
       </p>
     </div>

     <div class="grid-card">
       <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-6" width="64" height="64">
         <path stroke-linecap="round" stroke-linejoin="round" d="M14.25 6.087c0-.355.186-.676.401-.959.221-.29.349-.634.349-1.003 0-1.036-1.007-1.875-2.25-1.875s-2.25.84-2.25 1.875c0 .369.128.713.349 1.003.215.283.401.604.401.959v0a.64.64 0 0 1-.657.643 48.39 48.39 0 0 1-4.163-.3c.186 1.613.293 3.25.315 4.907a.656.656 0 0 1-.658.663v0c-.355 0-.676-.186-.959-.401a1.647 1.647 0 0 0-1.003-.349c-1.036 0-1.875 1.007-1.875 2.25s.84 2.25 1.875 2.25c.369 0 .713-.128 1.003-.349.283-.215.604-.401.959-.401v0c.31 0 .555.26.532.57a48.039 48.039 0 0 1-.642 5.056c1.518.19 3.058.309 4.616.354a.64.64 0 0 0 .657-.643v0c0-.355-.186-.676-.401-.959a1.647 1.647 0 0 1-.349-1.003c0-1.035 1.008-1.875 2.25-1.875 1.243 0 2.25.84 2.25 1.875 0 .369-.128.713-.349 1.003-.215.283-.4.604-.4.959v0c0 .333.277.599.61.58a48.1 48.1 0 0 0 5.427-.63 48.05 48.05 0 0 0 .582-4.717.532.532 0 0 0-.533-.57v0c-.355 0-.676.186-.959.401-.29.221-.634.349-1.003.349-1.035 0-1.875-1.007-1.875-2.25s.84-2.25 1.875-2.25c.37 0 .713.128 1.003.349.283.215.604.401.96.401v0a.656.656 0 0 0 .658-.663 48.422 48.422 0 0 0-.37-5.36c-1.886.342-3.81.574-5.766.689a.578.578 0 0 1-.61-.58v0Z" />
       </svg>
       <p>
         <strong> Extensible </strong> <br/>
         Additional basis functions can easily be added using the abstract base class of the basis functions.
       </p>
     </div>

     <div class="grid-card">
       <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="size-6" width="64" height="64">
         <path stroke-linecap="round" stroke-linejoin="round" d="M15.59 14.37a6 6 0 0 1-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 0 0 6.16-12.12A14.98 14.98 0 0 0 9.631 8.41m5.96 5.96a14.926 14.926 0 0 1-5.841 2.58m-.119-8.54a6 6 0 0 0-7.381 5.84h4.8m2.581-5.84a14.927 14.927 0 0 0-2.58 5.84m2.699 2.7c-.103.021-.207.041-.311.06a15.09 15.09 0 0 1-2.448-2.448 14.9 14.9 0 0 1 .06-.312m-2.24 2.39a4.493 4.493 0 0 0-1.757 4.306 4.493 4.493 0 0 0 4.306-1.758M16.5 9a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0Z" />
       </svg>
       <p>
         <strong> High performance </strong> <br/>
         Just-in-time compilation allows us to match and in some cases overcome the performance of SciPy's fortran based splines.
       </p>
     </div>

   </div>

   <h1 class="homepage-title">Comparison with SciPy</h1>

   <h2>Ease of use</h2>

   <p>
     A common task in image anlysis is to fit a closed spline of a given order with a
     fixed number of knots to a countour. Let's compare how we can achieve this task
     in SplineBox and SciPy.<br>
     <a href="./auto_examples/plot_splinebox_vs_scipy_coin.html">See full example →</a>
   </p>

.. container:: twocol

   .. container:: col

     .. raw:: html

       <p style="margin-right: 3em">
         In SplineBox all we need to do is select a basis function and specify
         the number of knots M and that the spline is closed in the spline constructor.
         The spline can then be fit to the data using its fit method.
       </p>

   .. container:: col

     .. code-block:: python

       import splinebox

       M = 10
       basis_function = splinebox.B3()

       spline = splinebox.Spline(
                    M, basis_function, closed=True
                )
       spline.fit(contour)

.. container:: twocol

   .. container:: col

     .. code-block:: python

       import scipy.interpolate

       M = 10
       N = len(data)
       k = 3

       t = np.arange(-k, M + k + 1) / M * N
       u = np.linspace(0, N, N, endpoint=True)

       tck, u = scipy.interpolate.splprep(
                 contour, u=u, k=k, task=-1,
                 s=0, t=t, per=N
             )

   .. container:: col

     .. raw:: html

       <p style="margin-left: 3em">
         Scipy requires you to pre-compute the parameter values for all knots and data points
         accounting for padding and periodicity of the data.
         This can be confusion and difficult to do.
       </p>

.. raw:: html

  <p style="margin-top: 3em">
    For additional examples comparing SplineBox to SciPy check out our example gallery.
    <br>
    <a href="./auto_examples/index.html">See examples →</a>
  </p>

  <h2>Performance</h2>

.. container:: twocol

  .. container:: col

    .. raw:: html

      <p>
        We compare the performance to splinbox to SciPy's splines
        on three main tasks:
        <ul>
        <li>Spline creation give a set of knots</li>
        <li>Evaluation of a spline at a given parameter value</li>
        <li>Data approximation using least-squares fitting</li>
        </ul>
        Splinebox out performs SciPy by approximately two orders of maginitued on the first two tasks and achives comparable performance for least-squares fitting of splines with 10 knots.
        <a href="./auto_examples/plot_performance_comparison_with_scipy.html">See detailed comparison →</a>
      </p>

  .. container:: col

    .. plot:: pyplots/plot_performance.py
