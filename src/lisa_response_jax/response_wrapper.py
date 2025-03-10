


class ResponseWrapper:
    """Wrapper to produce LISA TDI from TD waveforms

    This class takes a waveform generator that produces :math:`h_+ \pm ih_x`.
    (:code:`flip_hx` is used if the waveform produces :math:`h_+ - ih_x`).
    It takes the complex waveform in the SSB frame and produces the TDI channels
    according to settings chosen for :class:`pyResponseTDI`.

    The waveform generator must have :code:`kwargs` with :code:`T` for the observation
    time in years and :code:`dt` for the time step in seconds.

    Args:
        waveform_gen (obj): Function or class (with a :code:`__call__` function) that takes parameters and produces
            :math:`h_+ \pm h_x`.
        Tobs (double): Observation time in years.
        dt (double): Time between time samples in seconds. The inverse of the sampling frequency.
        index_lambda (int): The user will input parameters. The code will read these in
            with the :code:`*args` formalism producing a list. :code:`index_lambda`
            tells the class the index of the ecliptic longitude within this list of
            parameters.
        index_beta (int): The user will input parameters. The code will read these in
            with the :code:`*args` formalism producing a list. :code:`index_beta`
            tells the class the index of the ecliptic latitude (or ecliptic polar angle)
            within this list of parameters.
        t0 (double, optional): Start of returned waveform in seconds leaving ample time for garbage at
            the beginning of the waveform. It also removed the same amount from the end. (Default: 10000.0)
        flip_hx (bool, optional): If True, :code:`waveform_gen` produces :math:`h_+ - ih_x`.
            :class:`pyResponseTDI` takes :math:`h_+ + ih_x`, so this setting will
            multiply the cross polarization term out of the waveform generator by -1.
            (Default: :code:`False`)
        remove_sky_coords (bool, optional): If True, remove the sky coordinates from
            the :code:`*args` list. This should be set to True if the waveform
            generator does not take in the sky information. (Default: :code:`False`)
        is_ecliptic_latitude (bool, optional): If True, the latitudinal sky
            coordinate is the ecliptic latitude. If False, thes latitudinal sky
            coordinate is the polar angle. In this case, the code will
            convert it with :math:`\beta=\pi / 2 - \Theta`. (Default: :code:`True`)
        use_gpu (bool, optional): If True, use GPU. (Default: :code:`False`)
        remove_garbage (bool or str, optional): If True, it removes everything before ``t0``
            and after the end time - ``t0``. If ``str``, it must be ``"zero"``. If ``"zero"``,
            it will not remove the points, but set them to zero. This is ideal for PE. (Default: ``True``)
        n_overide (int, optional): If not ``None``, this will override the determination of
            the number of points, ``n``, from ``int(T/dt)`` to the ``n_overide``. This is used
            if there is an issue matching points between the waveform generator and the response
            model.
        orbits (:class:`Orbits`, optional): Orbits class from LISA Analysis Tools. Works with LISA Orbits
            outputs: `lisa-simulation.pages.in2p3.fr/orbits/ <https://lisa-simulation.pages.in2p3.fr/orbits/latest/>`_.
            (default: :class:`EqualArmlengthOrbits`)
        **kwargs (dict, optional): Keyword arguments passed to :class:`pyResponseTDI`.

    """

    def __init__(
        self,
        waveform_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=10000.0,
        flip_hx=False,
        remove_sky_coords=False,
        is_ecliptic_latitude=True,
        use_gpu=False,
        remove_garbage=True,
        n_overide=None,
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        **kwargs,
    ):

        # store all necessary information
        self.waveform_gen = waveform_gen
        self.index_lambda = index_lambda
        self.index_beta = index_beta
        self.dt = dt
        self.t0 = t0
        self.sampling_frequency = 1.0 / dt

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        assert isinstance(orbits, Orbits)

        if Tobs * YRSID_SI > orbits.t_base.max():
            warnings.warn(
                f"Tobs is larger than available orbital information time array. Reducing Tobs to {orbits.t_base.max()}"
            )
            Tobs = orbits.t_base.max() / YRSID_SI

        if n_overide is not None:
            if not isinstance(n_overide, int):
                raise ValueError("n_overide must be an integer if not None.")
            self.n = n_overide

        else:
            self.n = int(Tobs * YRSID_SI / dt)

        self.Tobs = self.n * dt
        self.is_ecliptic_latitude = is_ecliptic_latitude
        self.remove_sky_coords = remove_sky_coords
        self.flip_hx = flip_hx
        self.remove_garbage = remove_garbage

        # initialize response function class
        self.response_model = pyResponseTDI(
            self.sampling_frequency, self.n, orbits=orbits, use_gpu=use_gpu, **kwargs
        )

        self.use_gpu = use_gpu

        self.Tobs = (self.n * self.response_model.dt) / YRSID_SI

    @property
    def xp(self) -> object:
        return np if not self.use_gpu else cp


    def __call__(self, *args, **kwargs):
        """Run the waveform and response generation

        Args:
            *args (list): Arguments to the waveform generator. This must include
                the sky coordinates.
            **kwargs (dict): kwargs necessary for the waveform generator.

        Return:
            list: TDI Channels.

        """

        args = list(args)

        # get sky coords
        beta = args[self.index_beta]
        lam = args[self.index_lambda]

        # remove them from the list if waveform generator does not take them
        if self.remove_sky_coords:
            args.pop(self.index_beta)
            args.pop(self.index_lambda)

        # transform polar angle
        if not self.is_ecliptic_latitude:
            beta = np.pi / 2.0 - beta

        # add the new Tobs and dt info to the waveform generator kwargs
        kwargs["T"] = self.Tobs
        kwargs["dt"] = self.dt

        # get the waveform
        h = self.waveform_gen(*args, **kwargs)

        if self.flip_hx:
            h = h.real - 1j * h.imag

        self.response_model.get_projections(h, lam, beta, t0=self.t0)
        tdi_out = self.response_model.get_tdi_delays()

        out = list(tdi_out)
        if self.remove_garbage is True:  # bool
            for i in range(len(out)):
                out[i] = out[i][
                    self.response_model.tdi_start_ind : -self.response_model.tdi_start_ind
                ]

        elif isinstance(self.remove_garbage, str):  # bool
            if self.remove_garbage != "zero":
                raise ValueError("remove_garbage must be True, False, or 'zero'.")
            for i in range(len(out)):
                out[i][: self.response_model.tdi_start_ind] = 0.0
                out[i][-self.response_model.tdi_start_ind :] = 0.0

        return out