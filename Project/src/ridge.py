git filter-branch --commit-filter '
    if [ "$GIT_COMMITTER_NAME" = "Saumya Rawat" ];
    then
            GIT_COMMITTER_NAME="Saumya Rawat";
            GIT_AUTHOR_NAME="Saumya Rawat";
            GIT_COMMITTER_EMAIL="saumya.rawat25@gmail.com";
            GIT_AUTHOR_EMAIL="saumya.rawat25@gmail.com";
            git commit-tree "$@";
    else
            git commit-tree "$@";
    fi' HEAD