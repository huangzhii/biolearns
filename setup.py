# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Mon Feb 10 17:57:08 2020
# Author: Zhi Huang, Purdue University
#
# The original code came with the following disclaimer:
#
# This software is provided "as-is".  There are no expressed or implied
# warranties of any kind, including, but not limited to, the warranties
# of merchantability and fitness for a given application.  In no event
# shall Zhi Huang be liable for any direct, indirect, incidental,
# special, exemplary or consequential damages (including, but not limited
# to, loss of use, data or profits, or business interruption) however
# caused and on any theory of liability, whether in contract, strict
# liability or tort (including negligence or otherwise) arising in any way
# out of the use of this software, even if advised of the possibility of
# such damage.

def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.6',
                                 'Programming Language :: Python :: 3.7',
                                 'Programming Language :: Python :: 3.8',
                                 ('Programming Language :: Python :: '
                                  'Implementation :: CPython'),
                                 ('Programming Language :: Python :: '
                                  'Implementation :: PyPy')
                                 ],
                    cmdclass=cmdclass,
                    python_requires=">=3.6",
                    install_requires=[
                        'numpy>={}'.format(NUMPY_MIN_VERSION),
                        'scipy>={}'.format(SCIPY_MIN_VERSION),
                        'joblib>={}'.format(JOBLIB_MIN_VERSION)
                    ],
                    package_data={'': ['*.pxd']},
                    **extra_setuptools_args)

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    'dist_info',
                                                    '--version',
                                                    'clean'))):
        # For these actions, NumPy is not required
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scikit-learn when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION
    else:
        if sys.version_info < (3, 6):
            raise RuntimeError(
                "BioLearns requires Python 3.6 or later. The current"
                " Python version is %s installed in %s."
                % (platform.python_version(), sys.executable))

        check_package_status('numpy', NUMPY_MIN_VERSION)

        check_package_status('scipy', SCIPY_MIN_VERSION)

        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
