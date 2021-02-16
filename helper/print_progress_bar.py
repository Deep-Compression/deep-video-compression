def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', decimals=2, length=50, fill='█',
                       end='\r'):
    """
        Prints a progress bar to console.

        :param iteration: Current iteration
        :param total: Total iterations
        :param prefix: Prefix string
        :param suffix: Suffix string
        :param decimals: Number of decimals in percent complete
        :param length: Character length
        :param fill: Bar fill character
        :param end: End character (e.g. '\r', '\r\n')
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=end)

    if iteration == total:
        print()
