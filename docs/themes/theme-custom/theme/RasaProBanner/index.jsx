import * as React from 'react';
import clsx from 'clsx';

import styles from './styles.module.css';

function RasaProBanner({isLoading, ...props}) {
  return (
    <>
      <div class="mdx-box admonition admonition-tip alert alert--success">
        <div class="mdx-box admonition-heading">
          <h5>
            <span class="admonition-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="16" viewBox="0 0 12 16">
                <path fill-rule="evenodd" d="M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z">
                </path>
              </svg>
            </span>Rasa Pro License
          </h5>
        </div>
      <div class="mdx-box admonition-content">
        <p>You'll need a license to get started with Rasa Pro. 
        {' '}
          <a href="https://rasa.com/connect-with-rasa/" target="_blank" rel="noopener noreferrer">
            Talk with Sales
          </a>
        </p>
      </div>
      </div>
  </>
  )
}

export default RasaProBanner;
