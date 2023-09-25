import os
import subprocess
import click

def git_rebase_main_or_master(repo_path):
    try:
        # Change directory to the repository path
        os.chdir(repo_path)

        # Check if the repository has the main branch
        if os.system('git show-ref --quiet refs/heads/main') == 0:
            branch_name = 'main'
        # Otherwise, check for the master branch
        elif os.system('git show-ref --quiet refs/heads/master') == 0:
            branch_name = 'master'
        else:
            print(f"Neither 'main' nor 'master' branch found in {repo_path}. Skipping...")
            return

        print(f"Rebasing '{branch_name}' branch in {repo_path}...")
        subprocess.run(['git', 'checkout', branch_name])
        subprocess.run(['git', 'pull', '--rebase'])

        print(f"Rebasing completed in {repo_path}")
    except Exception as e:
        print(f"Error while processing {repo_path}: {str(e)}")

@click.command()
@click.argument('folder_path', type=click.Path(exists=True))
def main(folder_path):
    """
    Rebase the 'main' or 'master' branch in all git repositories inside the specified folder.
    """
    git_rebase_main_or_master(os.getcwd())
    
    for item in os.listdir(folder_path):
        repo_path = os.path.join(folder_path, item)
        print(f"Loking: {repo_path}")    
        if os.path.isdir(repo_path):
            git_rebase_main_or_master(repo_path)

if __name__ == "__main__":
    main()